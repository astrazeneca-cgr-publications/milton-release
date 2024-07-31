import logging
from dataclasses import asdict
from functools import cached_property
from pathlib import Path

import pandas as pd
import numpy as np
# reporting
from bokeh import __version__ as bokeh_version
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer, roc_curve, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_validate
from boruta import BorutaPy
from xgboost import XGBClassifier
import pickle
from typing import Optional, Dict, List, Union

from .random import new_rnd, set_random_state, randint
from .classification import (MiltonException, MiltonPipeline, DataTransformer,
                             specificity)
from .configuration import ModelType
from .data_desc import Col, Fld
from .data_info import OLINK, UKB
from .data_source import DfBag, UkbDataStore
from .patsel import UkbPatientSelector
from .processing import poststratify
from .plotting import (Proj2dPlots, html_components, hv_render_html,
                       plot_feature_importance,
                       plot_multi_curve,
                       roc_markers)
from .qvsig import CollapsingAnalyser
from .utils import ensure_gender_proportions, RowList, first_non_null_datetime


def make_template_env():
    milton_root = Path(__file__).parent.parent.absolute()
    return Environment(
        loader=FileSystemLoader(str(milton_root / 'resources')),
        cache_size=0,  # makes debugging easier
        autoescape=select_autoescape(['html', 'xml']))


def mean_shift(df, group_by, abs=True):
    """(Absolute) and standardized difference between group means.
    """
    ms = df.groupby(group_by)\
        .agg(['mean', 'std']).T\
        .swaplevel(0, 1)\
        .pipe(lambda _df: (_df.loc['mean', 1] - _df.loc['mean', 0]) / df.std())
    ms = ms.abs() if abs else ms
    return ms.sort_values()


def cat_distribution(series, by, relative=True):
    df = series.groupby([by, series])\
        .size()\
        .unstack(level=0)
    
    return df / df.sum() if relative else df


def cat_distribution_chi2(series, by):
    return series.groupby([by, series])\
        .size()\
        .unstack(level=0)\
        .dropna()\
        .pipe(stats.chi2_contingency)[1]
        
        
def running_ncr(scores, cases, min_score):
    """Novel Case Ratio (NCR) for each sample, sorted by decreasing score.
    """
    is_case = pd.Series(True, cases).reindex(scores.index, fill_value=False)
    s = scores.mask(is_case, 1)\
        .loc[lambda s: s >= min_score]\
        .sort_values(ascending=False)
    return pd.Series(np.arange(1, len(s) + 1) / is_case.sum(), 
                        index=s.index)


class Evaluator:
    DATASETS = [UKB, OLINK]  # supported datasets
    
    def __init__(self, settings):
        self.settings = settings.copy()
        self._templates = make_template_env()
        self._conf = settings()
        self.dst = UkbDataStore(**self._conf.dataset.asdict())
        self._dd = self.dst.ukb.data_dict
        self._sel = UkbPatientSelector(
            data_store=self.dst,
            ukb_controls_cases_ratio=self._conf.patients.controls_to_cases_ratio,
            sampling_factors=self._conf.patients.sampling_factors,
            infrequent_gender_thresh=self._conf.patients.infrequent_gender_thresh,
            diag_fields=self._conf.patients.diag_fields,
            use_icd10_extras=False,  # use exclusively UKB so we can match dates
            controls=self._conf.patients.training_controls)
        self._models = {}
        self.selected_features = None
        self.held_out = None
        self.best_hyper_params = {}
        self._load_aux_data()
        
    def _load_aux_data(self):
        # gender info for all subjects
        self.ukb_genders = self.dst.ukb.read_processed(
            self.dst.ukb.to_schema([Fld.GENDER]))\
            .loc[:, Col.GENDER]\
            .dropna()
        
    @property
    def n_replicas(self):
        return self._conf.analysis.n_replicas
    
    @property
    def model_type(self):
        return self._conf.analysis.model_type
    
    @property
    def needs_case_exclusions(self):
        return (self.model_type != ModelType.STANDARD 
                or self._conf.analysis.diagnosis_cap is not None)
    
    @property
    def main_cohort(self):
        return self.cohort_replicas[0]
    
    @cached_property
    def rnd(self) -> np.random.Generator:
        """Signle random state derived from the configuration, used throughout
        the lifetime of the pipeline.
        """
        # using string representation of configuration since it prints nicely
        # everything. 
        return new_rnd(str(self._conf))
    
    def run(self, *, 
            cohort: Optional[Union[pd.Series, pd.Index]] = None, 
            features: Optional[Dict[str, List[str]]] = None, 
            data: Optional[pd.DataFrame] = None,
            fetch_only=False):
        if self.is_fitted:
            raise ValueError('Evaluation cannot be re-run.')
        
        if features is not None:
            self._schema = self._get_custom_schema(features)
        elif data is None:
            self._schema = self.get_config_schema()
        else:
            # ensure presence of AGE, GENDER and UKB index
            self._check_data(data)
            self._schema = None
        
        if self._schema:
            self.data_index = self.get_data_index(self._schema)
        else: 
            self.data_index = data.index
        
        if cohort is None:
            spec = self._conf.patients.spec
        else:
            self._check_cohort(cohort)
            spec = cohort.loc[cohort.index.intersection(self.data_index)]
        
        self.custom_data = data
        with set_random_state(self.rnd):
            select_features = self._conf.feature_selection.iterations > 0
            held_out = self._conf.analysis.held_out_frac
            if select_features and held_out is not None:
                raise MiltonException(
                    'Evaluation on held-out set is not possible with feature '
                    'selection')
            if select_features:
                # feature selection uses a fraction of available data that must 
                # be excluded from model training
                case_exclusions, ctl_exclusions = self._select_features(spec)
                # the remaining data is used to perform hyper-parameter tuning
                # and model evaluation
                self._fetch_training_cohorts(
                    spec, case_exclusions, ctl_exclusions,
                    # feature selection is done on 50% of data
                    min_case_thresh = self._conf.analysis.min_cases // 2)
                self._fetch_training_data()
                # hyper param tuning and evaluation done on the same data
                self._select_estimator()
                self._evaluate()
            # After feature selection, use *all* available data (cases) to fit
            # the final models. 
            self._fetch_training_cohorts(spec)
            self._fetch_training_data()
            if not fetch_only:
                if not select_features:
                    if held_out is not None:
                        held_out_cohort = self.full_cohort.sample(frac=held_out)
                        self.held_out = held_out_cohort.index
                        logging.info(f'HELD-OUT set of {len(self.held_out)}.')
                    self._select_estimator()
                    self._evaluate()
                self._fit_all_models()
        return self
    
    def _select_features(self, spec):
        # extract balanced cohorts with 1:1 cases to controls
        logging.info('Running feature selection')
        with self._sel.options(ukb_controls_cases_ratio=1):
            cohorts, _ = self._do_fetch_cohorts(
                spec,
                case_frac=self._conf.feature_selection.data_frac,
                n_replicas=self._conf.feature_selection.iterations)
        full_cohort = pd.concat(cohorts).groupby(level=0).max()
        dt = DataTransformer(**self._conf.preproc.asdict())
        data = self.load_data(rows=full_cohort.index).pipe(dt.fit_transform)
        all_features = set(self._conf.feature_selection.preserved)
        for i, y in enumerate(cohorts):
            X = data.loc[y.index]
            features = self._do_select_features(X, y)
            logging.info(
                f'Iteration {i} of feature selection selected: {sorted(features)}')
            all_features |= features
        if len(all_features) == 2:
            raise MiltonException(
                'Feature selection algorithm failed to select features.')
        self.selected_features = sorted(all_features)
        logging.info(f'Final feature selection: {self.selected_features}')
        cases = full_cohort.index[full_cohort == True]
        controls = full_cohort.index[full_cohort == False]
        return cases, controls
    
    def _do_select_features(self, X, y):
        clf = RandomForestClassifier(
            class_weight='balanced', max_depth=5, n_jobs=-1, 
            random_state=randint())
        bor = BorutaPy(clf, n_estimators='auto', random_state=randint())
        bor.fit(X.to_numpy(), y)
        features = set()
        confirmed = bor.ranking_ == 1
        tentative = bor.ranking_ == 2
        if confirmed.any():
            features |= set(X.columns[confirmed])
        if self._conf.feature_selection.tentative and tentative.any():
            features |= set(X.columns[tentative])
        return features
    
    def _fetch_training_cohorts(self, spec, 
                                case_exclusions=None, 
                                ctl_exclusions=None,
                                min_case_thresh=None):
        cohorts, self.all_cases = self._do_fetch_cohorts(
            spec, 
            case_exclusions=case_exclusions, 
            ctl_exclusions=ctl_exclusions,
            n_replicas=self.n_replicas)
        self.cohort_replicas = [c.dropna().astype('int') for c in cohorts]
        self.full_cohort = pd.concat(cohorts).groupby(level=0).max().astype('int') 
        logging.info(
            f'Fetched {len(self.cohort_replicas)} training cohorts with '
            f'{self.full_cohort.sum()} total cases and {len(self.all_cases)} '
            'known cases.')
        num_cases = self.full_cohort.sum()
        if min_case_thresh is None:
            min_case_thresh = self._conf.analysis.min_cases
        if num_cases < min_case_thresh:
            raise MiltonException(
                f'Not enough cases for training: {num_cases}. Threshold is: '
                f'{min_case_thresh}.')
    
    def _do_fetch_cohorts(self, spec, 
                          case_exclusions: pd.Index = None, 
                          ctl_exclusions: pd.Index = None, 
                          n_replicas: int = 1,
                          case_frac: float = None):
        """Core cohort selection logic.
        """
        is_custom_cohort = isinstance(spec, pd.Series)
        ukb_subjects = self._sel.ukb_subjects
        data_index = self.data_index
        used_subjects = self._conf.patients.used_subjects
        if used_subjects is not None:
            data_index = data_index.intersection(used_subjects)
        if case_exclusions is None:
            case_exclusions = pd.Index([], 'int64')
        if ctl_exclusions is None:
            ctl_exclusions = pd.Index([], 'int64')
        case_exclusions = case_exclusions.union(ukb_subjects.difference(data_index))
        ctl_exclusions = ctl_exclusions.union(case_exclusions)
        if is_custom_cohort:
            case_spec = spec[spec == True].index
            ctl_spec = spec[spec == False].index
            if not ctl_spec.empty:
                # there are specific controls to be used: add all non-controls
                ctl_exclusions = ctl_exclusions.union(data_index.difference(ctl_spec))
        else:
            case_spec = spec  # ICD10 codes
        if self.needs_case_exclusions:
            # need to discard some cases 
            if is_custom_cohort:
                raise MiltonException(
                    'Time-based model types cannot be used with a custom cohort'
                    ' since there are no available diagnoses.')
            cases_to_drop = self._fetch_excluded_cases(case_spec)
            case_exclusions = case_exclusions.union(cases_to_drop)
        cohorts, all_cases = self._sel(
            case_spec, 
            add_controls=n_replicas, 
            case_exclusions=case_exclusions,
            # make sure the excluded cases do not become controls
            ctl_exclusions=ctl_exclusions,
            drop_same_chapter_ctl=self._conf.patients.drop_same_chapter_controls,
            fraction=case_frac,
            ret_known_cases=True)
        num_cases = int(len(all_cases) / (case_frac or 1.0))
        if self.needs_case_exclusions:
            removed_frac = np.round(
                (len(cases_to_drop) / num_cases) * 100, 2)
            diag_cap = self._conf.analysis.diagnosis_cap
            logging.info(
                f'Special model condition (type: {self.model_type}, diagnosis '
                f'cap: {diag_cap}) resulted in removal of '
                f'{len(cases_to_drop)} which is {removed_frac}% of all matched')
        return cohorts, all_cases
    
    def _fetch_excluded_cases(self, spec):
        """Returns the set of cases matching the specification but which need to
        be excluded from the cohort due to unfulfilled requirements of the 
        prognostic/diagnostic model types.
        """
        diag_dates = self.earliest_diagnosis_dates(spec)
        cases = diag_dates.index
        if not diag_dates.empty:
            to_keep = pd.Series(True, cases)
            if self._conf.analysis.diagnosis_cap is not None:
                cutoff = pd.to_datetime(self._conf.analysis.diagnosis_cap)
                to_keep &= diag_dates < cutoff
            if self.model_type != ModelType.STANDARD:
                biom_dates = self.biomarker_diagnosis_dates(cases).reindex(cases)
                if self.model_type == ModelType.PROGNOSTIC:
                    to_keep &= biom_dates < diag_dates
                else:
                    # diagnostic model
                    to_keep &= biom_dates >= diag_dates
                max_lag = self._conf.analysis.max_time_lag * 365
                year_cap = (biom_dates - diag_dates).dt.days.abs() <= max_lag
                to_keep &= year_cap
            # return cases to *exclude*
            return cases[~to_keep]
        else:
            return pd.Index([], dtype='int')
    
    def biomarker_diagnosis_dates(self, cases=None):
        """Returns the first non-null value (the earliest) of blood and urine 
        sample sign-off timestamps.
        """
        return self.dst.ukb.read_data(
            self.dst.ukb.to_schema([21841, 21842]), rows=cases)\
            .concat()\
            .compute()\
            .pipe(first_non_null_datetime)
    
    def earliest_diagnosis_dates(self, codes=None):
        parts = []
        codes = list(codes) if codes is not None else []
        diag_fields = self._conf.patients.diag_fields
        for diag_field in map(str, diag_fields):
            # bolean mask selecting values matching one of the codes
            # constrained to rows with at least one matching value
            data = self.dst.ukb.read_data(self.dst.ukb.to_schema([diag_field]))
            if codes:
                select_codes = lambda s: s.isin(codes)
            else:
                # no codes means take all diagnoses
                select_codes = lambda s: s.notna()
            data = data.map(lambda df: df.apply(select_codes))
            diag_mask = data.map(lambda df: df.loc[df.max(axis=1)])\
                .concat()\
                .compute()
            # death causes and dates need to be treated bit differently
            if diag_field in ['40001', '40002']:
                # collapse the mask to a single column corresponding to the
                # single date of death
                diag_mask = diag_mask.max(axis=1).to_frame()
            if not diag_mask.empty:
                date_field = self.dst.DATESOF_MAP['ukb'][diag_field]
                diag_dates = self.dst.ukb.read_data(
                    self.dst.ukb.to_schema([date_field]), 
                    rows=diag_mask.index)\
                    .concat()\
                    .compute()\
                    .reindex(diag_mask.index)
                if date_field == '40000':
                    # date of death: use the first instance - a single column
                    diag_dates = diag_dates[['40000-0.0']]
                earliest_date = RowList(diag_dates, diag_mask).min()
                parts.append(earliest_date)  
        dates = pd.concat(parts, axis=1)
        return RowList(dates, dates.notna()).min() # earliest of diagnoses
    
    def load_data(self, rows=None, features=None):
        """Unified interface to data loading for custom data or one or many
        datasets. 
        """
        if self.custom_data is not None:
            everything = slice(None)
            rows = rows if rows is not None else everything
            features = features if features is not None else everything
            return self.custom_data.loc[rows, features]
        else:
            parts = []
            all_features = pd.Index([], dtype='O')
            for name, schema in self._schema.items():
                if name == UKB:
                    if features is not None:
                        schema = schema[schema.index.isin(features)]
                        if schema.empty:
                            continue
                    df = self.dst.ukb.read_processed(
                        schema,
                        rows=rows,
                        derived=self._conf.analysis.derived_biomarkers)
                else:
                    if features is not None:
                        schema = sorted(set(schema) & set(features))
                        if not schema:
                            continue
                    df = self.dst[name].load(schema, rows=rows)\
                        .concat().compute()
                name_clash = all_features.intersection(df.columns)
                if not name_clash.empty:
                    df.rename(columns={
                        col: f'{col}:{name}' for col in name_clash
                    }, inplace=True)
                all_features = all_features.append(df.columns)
                parts.append(df)
            return pd.concat(parts, axis=1)
    
    def _fetch_training_data(self):
        data = self.load_data(self.full_cohort.index, self.selected_features)
        logging.info(f'Fetched training data of shape: {data.shape}.')
        if not data.columns.is_unique:
            dupl = sorted(data.columns[data.columns.duplicated()].unique())
            raise MiltonException(
                f'Duplicate columns found in final data schema: {dupl}')
        self.data = data
    
    def _select_estimator(self):
        X0, y0 = self.Xy(0, exclude=self.held_out)
        if self.hyper_params:
            logging.info('Tuning hyper parameters...')
            self.estimator_ = self._select_estimator_with_hpt(X0, y0)
            best_params = ', '.join(f'{p}={v}' 
                                    for p, v in self.best_hyper_params.items())
            logging.info(f'Best hyper parameters: {best_params}')
        else:
            logging.info('Using default estimator')
            self.estimator_ = self._default_estimator()
        
    def _fit_all_models(self):
        logging.info(f'Fitting {self.n_replicas} model replicas.')
        X0, y0 = self.Xy(0)
        self.clf_replicas = []
        for i in range(self.n_replicas):
            clf = self._pipeline().fit(*self.Xy(i))
            self.clf_replicas.append(clf)
        self._models['General'] = self.clf_replicas[0]
        
        logging.info('Fitting supplementary models.')
        genders = X0[Col.GENDER].unique().dropna()
        for gender in genders:
            mdl = self._fit_model(X0, y0, gender=gender)
            if mdl:
                self._models[gender] = mdl
        # those models will syncronize on the main model
        # it shouldn't affect performance a lot since they're ligthweight
        self._models['General-LR'] = self._fit_model(X0, y0, lr=True)
        for gender in genders:
            mdl = self._fit_model(X0, y0, gender=gender, lr=True)
            if mdl:
                self._models[gender + '-LR'] = mdl
            
    def _reorder_features(self, X):
        """Reorders the features of X in the order of importance according 
        to the main classifier. This will help LR-based models with correlated
        feature removal - of each correlated pair the less important will be 
        removed.
        """
        most_important = self.feature_importance.index
        common = most_important.intersection(X.columns)
        other = X.columns.difference(most_important)
        new_order = common.to_list() + other.to_list()
        return X.loc[:, new_order]
        
    def _fit_model(self, X, y, gender=None, lr=False):
        params = asdict(self._conf.preproc)
        if lr:
            # feature reordering will synchronize on the main model
            # parallelism is lost here
            X = self._reorder_features(X)
            clf = LogisticRegression()
            params['drop_correlated'] = self._conf.analysis.correlation_thresh
        else:
            clf = self.estimator_
            
        if gender is not None:
            X = X[X[Col.GENDER] == gender]
            y = y.reindex(X.index)
            num_cases = y.sum()
            if num_cases < self._conf.analysis.min_cases:
                logging.warning('Not enough data for fitting gender-specific '
                                f'model for {gender}. Num cases: {num_cases}')
                return None
        return self._pipeline(clf).fit(X, y)
    
    def Xy(self, replica_n, exclude=None):
        """Returns a pair of training data and labels for cohort replica n.
        """
        y = self.cohort_replicas[replica_n]
        if exclude is not None:
            y = y[y.index.difference(exclude)]
        X = self.data.loc[y.index]
        return X, y
        
    def _evaluate(self):
        cv_metrics = []
        roc_curves = []
        pr_curves = []
        logging.info('Evaluating model performance')
        if self._conf.analysis.evaluate_all_replicas:
            n_eval = self.n_replicas
        else:
            # when you have no time to waste
            n_eval = 1
        for i in range(n_eval):
            metrics, roc, pr = self._eval_replica(i)
            cv_metrics.append(metrics)
            roc_curves.extend(roc)
            pr_curves.extend(pr)
        self.replica_metrics = pd.DataFrame(cv_metrics)
        self.roc_curves = np.stack(roc_curves, axis=0)
        self.pr_curves = np.stack(pr_curves, axis=0)
            
    def _eval_replica(self, replica_n, roc_n_points=20):
        X, y = self.Xy(replica_n)
        clf = self._pipeline()
        if self.held_out is None:
            cv_splits = list(StratifiedKFold(5, shuffle=True).split(X, y))
        else:
            # single CV split
            held_out_mask = X.index.isin(self.held_out)
            cv_splits = [(
                np.flatnonzero(~held_out_mask), 
                np.flatnonzero(held_out_mask)
            )]
        res = cross_validate(
            clf, X, y, 
            cv=cv_splits, 
            scoring={
                'auc': get_scorer('roc_auc'),
                'f1': get_scorer('f1'),
                'precision': get_scorer('precision'), 
                'sensitivity': get_scorer('recall'), 
                'avg_prec': get_scorer('average_precision'),
                'specificity': specificity
            },
            return_estimator=True)
        cv_metrics = {
            'F1': res['test_f1'].mean(),
            'AUC': res['test_auc'].mean(),
            'Sensitivity': res['test_sensitivity'].mean(),
            'Specificity': res['test_specificity'].mean(),
            'Precision': res['test_precision'].mean(),
            'AveragePrecision': res['test_avg_prec'].mean()
        }
        roc = []
        pr = []
        for i, (_, ix_val) in enumerate(cv_splits):
            y_val = y.iloc[ix_val]
            X_val = X.iloc[ix_val]
            y_prob = res['estimator'][i].predict_proba(X_val)
            for curves, func in ((roc, roc_curve), 
                                 (pr, precision_recall_curve)):
                cx, cy, _ = func(y_val, y_prob) 
                point_ix = roc_markers(cx, cy, n=roc_n_points)
                cxy = np.vstack((cx[point_ix], cy[point_ix]))
                curves.append(cxy.T)
        return cv_metrics, roc, pr
    
    @property
    def estimator_type(self):
        return self._conf.analysis.default_model
        
    def _default_estimator(self):
        name = self.estimator_type
        if name == 'xgb':
            return XGBClassifier(
                # defaults, based on extensive tuning
                reg_alpha=0.5,
                n_estimators=200,
                min_child_weight=1,
                max_depth=5,
                learning_rate=0.05,
                gamma=0.7,
                # default options
                eval_metric='logloss',
                random_state=randint(),
                n_jobs=4)
        elif name == 'random-forest':
            return RandomForestClassifier(
                n_jobs=4,
                random_state=randint(),
            )
        elif name == 'lr':
            return LogisticRegression(solver='newton-cg', max_iter=200)
        elif name == 'autoencoder':
            from scikeras.wrappers import KerasClassifier
            import keras
            input_shape = self.data.shape[1]
            def model0():
                return keras.Sequential([
                    keras.Input(shape=input_shape),
                    keras.layers.Dropout(.15),
                    keras.layers.Dense(8, activation='relu'),
                    keras.layers.Dropout(.15),
                    keras.layers.Dense(4, activation='relu'),
                    keras.layers.Dropout(.15),
                    keras.layers.Dense(8, activation='relu'),
                    keras.layers.Dense(1, activation="sigmoid", 
                                       kernel_initializer='normal'),
                ])
            return KerasClassifier(
                model0, 
                optimizer='adam', 
                loss='binary_crossentropy',
                epochs=20,
                verbose=0)
        else:
            raise MiltonException(
                f'Unsupported classifier in configuration: {name}')
            
    @property
    def hyper_params(self):
        return self._conf.analysis.hyper_parameters
            
    def _select_estimator_with_hpt(self, X, y):
        hpt_spec = (self.estimator_type, 
                    self._default_estimator(), 
                    self.hyper_params)
        params = asdict(self._conf.preproc)
        mdl = MiltonPipeline(
            [hpt_spec], params, 
            grid_search=True,
            selection_metric=self._conf.analysis.hyper_param_metric)
        mdl.fit(X, y)
        self.best_hyper_params = {
            hparam: getattr(mdl.estimator_, hparam)
            for hparam in self.hyper_params
        }
        return mdl.estimator_
        
    def _pipeline(self, estimator=None):
        return MiltonPipeline(
            estimator or self.estimator_, 
            asdict(self._conf.preproc))
        
    @cached_property
    def validation_metrics(self):
        self._check_if_fitted()
        return self.replica_metrics.mean()
    
    @cached_property
    def data_metrics(self):
        self._check_if_fitted()
        recs = []
        n_features = self.data.shape[1]
        for cohort in self.cohort_replicas:
            recs.append((n_features, len(cohort), cohort.sum()))
        return pd.DataFrame(recs, 
                            columns=['n_features', 'train_size', 'train_cases'])
    
    @cached_property
    def qv_subjects(self):
        return pd.read_csv(self._conf.analysis.qv_subject_subset)\
            .pipe(lambda df: pd.Index(df['eid']))
            
    def for_collapsing(self, cohort: Union[pd.Series, pd.Index]):
        """Reindexes the cohort to be suitable for running the collapsing
        analysis on it.
        """
        if isinstance(cohort, pd.Index):
            cohort = pd.Series(1, index=cohort)
        else:
            cohort = cohort.astype('int64')
        # Extend the cohort with as many controls as possible
        if self._conf.analysis.qv_subject_subset:
            new_index = self.qv_subjects.intersection(self.ukb_genders.index)
        else:
            new_index = self.ukb_genders.index
        if self._conf.analysis.collapsing_on_data_index:
            # for scenarios in which data_index is smaller than the full UKB
            # (eg OLINK). By default this option is off leading to higher
            # statistical power (larger cohort size)
            new_index = new_index.intersection(self.data_index)
        cohort_all_ctl = cohort.reindex(new_index, fill_value=0)
        if self._conf.patients.collapsing_controls is not None:
            # exclude specific controls if needed
            ctl_ix = pd.Index(self._conf.patients.collapsing_controls).unique()
            ctl_to_drop = cohort_all_ctl.index[cohort_all_ctl == 0].difference(ctl_ix)
            print(f'Dropping {len(ctl_to_drop)} controls due to collapsing ctl spec')
            cohort_all_ctl.drop(ctl_to_drop, inplace=True)
        # and then resample the controls to match cases in gender distribution
        counts = cohort_all_ctl.value_counts()
        if len(counts) < 2:
            raise ValueError(
                'Cannot perform collapsing analysis due to missing cases or '
                f'controls: {counts.to_dict()}')
        return ensure_gender_proportions(
            cohort_all_ctl, self.ukb_genders,
            case_drop_cost=100)  # don't downsample cases at all
    
    @cached_property
    def qv_analyser(self):
        return CollapsingAnalyser(
            Path(self._conf.dataset.location) 
            / self._conf.analysis.qv_model_dir)
    
    def qv_significance(self, cohort, pval_thresh=None):
        """Calculates rare variant enrichment for all available QV models.
        This is a pandas DataFrame with QV model name as first level of index,
        gene name as second level. Note that the procedure adds its own controls
        
        Parameters
        ----------
        cohort : pd.Series, 1s and 0s.
          Patient cohort to run the analysis for. When a series, it must be of 
          integer type where 1s are cases and 0s are controls. When pd.Index, 
          it defines IDs of the cases.
          
        pval_thresh : float, optional
          Maximum p-value to report. When None, use the value specified in the 
          configuration.
        """
        if pval_thresh is None:
            pval_thresh = self._conf.analysis.gene_sign_thresh
        cohort = self.for_collapsing(cohort)
        if not cohort.index.is_unique:
            print('----------Non-unique cohort encountered:---------')
            print('Check it out under Evaluator.strange_cohort attribute')
            print('-------------------------------------------------')
            self.strange_cohort = cohort
            cohort = cohort[~cohort.index.duplicated()]
        return DfBag(self.qv_analyser.mdl_names)\
            .map(self.qv_analyser, cohort, pval_thresh=pval_thresh)\
            .concat(keys=self.qv_analyser.mdl_names)\
            .compute()\
            .rename_axis(index=['QV model', 'gene'])
    
    def _repr_html_(self):
        if self.is_fitted:
            y = self.main_cohort
            info = pd.Series([
                self.data.shape[1],
                y.shape[0],
                y.mean()], 
                index=['# features', 'Cohort size', 'Case ratio'])
            
            full_info = pd.concat([self.validation_metrics, info])\
                .to_frame('')\
                .to_html(float_format=lambda s: '%.2f' % s)
            
            features = self.feature_importance\
                .iloc[:7]\
                .mean(axis=1)\
                .to_frame('')\
                .to_html(float_format=lambda s: '%.2f' % s)
            
            template = '''
                <table>
                <tr>
                    <th>Summary</th> 
                    <th>Most Important Features</th>
                </tr>
                <tr>
                    <td>{full_info}</td>
                    <td>{features}</td>
                </tr>
                </table>
                '''
            return template.format(
                full_info=full_info, 
                features=features)
        else:
            return 'Model not fitted yet'
        
    def _check_data(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError('Custom data must be a pandas DataFrame object.')
        if df.columns.duplicated().any() or df.index.duplicated().any():
            raise ValueError(
                'Custom data must have unique index/column axes.')
        missing_cols = set([Col.AGE, Col.GENDER]).difference(df.columns)
        if missing_cols:
            raise ValueError(f'Missing columns in custom data: {missing_cols}')
        unrecognized = df.index.difference(self.ukb_genders.index)
        if not unrecognized.empty:
            raise ValueError(
                f'Found {len(unrecognized)} unrecognized UKB IDs in index')
        if df.index.duplicated().any():
            raise ValueError('Custom data index contains duplicates.')
        
    def _get_custom_schema(self, features: Dict[str, List[str]]):
        unsupported = set(features).difference(self.DATASETS)
        if unsupported:
            raise MiltonException('Unsupported datasets: ', unsupported)
        schema = {}
        if UKB not in features:
            features |= {UKB: []}
        for ds_name, ds_features in features.items():
            ds_features = [str(f) for f in ds_features]
            if ds_name == UKB:
                for required in [Fld.AGE, Fld.GENDER]:
                    if required not in ds_features:
                        ds_features.append(required)
                schema[ds_name] = self.dst.ukb.to_schema(ds_features)
            else:
                schema[ds_name] = list(ds_features)
        return schema
    
    def get_config_schema(self):
        ftconf = self._conf.features.asdict()
        schema = {}
        if ftconf['olink'] is True:
            olink = self.dst[OLINK].dataset
            schema[OLINK] = [c for c in olink.schema if c != olink.index_col]
        elif ftconf['olink']:
            # expecting list of feature names
            schema[OLINK] = list(ftconf['olink'])
        del ftconf['olink']
        if ftconf.get('ukb_custom', []):
            ukb_schema = self.dst.ukb.to_schema(ftconf['ukb_custom'])
        else:
            ukb_schema = pd.Series([])
        del ftconf['ukb_custom']
        schema[UKB] = pd.concat([
            self._dd.predefined(**ftconf),  # includes AGE and GENDER
            ukb_schema])
        return schema
    
    def get_data_index(self, schema):
        """Intersection of indices of all datasets.
        """
        data_index = None
        for dataset in schema:
            new_ix = self.dst[dataset].dataset.index
            if data_index is None:
                data_index = new_ix
            else:
                data_index = data_index.intersection(new_ix)
        return data_index
    
    def _check_cohort(self, cohort):
        if not isinstance(cohort, pd.Series):
            raise ValueError('Custom cohort must be a pandas Series object.')
        if cohort.dtype != 'bool':
            raise ValueError('Custom cohort must be of boolean type.')
        if cohort.index.dtype != 'int':
            raise ValueError('Custom cohort index must be of integer type.')
        if cohort.isna().any() or cohort.index.isna().any():
            raise ValueError('Custom cohort must not have null values.')
        if not cohort.index.is_unique:
            raise ValueError('Custom cohort index must not have duplicates.')
        unknown = len(cohort.index.difference(self.data_index))
        if unknown:
            raise ValueError(
                f'Found {unknown} unrecognized UKB IDs in custom cohort index.')
        
    @property    
    def is_fitted(self):
        return hasattr(self, 'clf_replicas')
    
    def _check_if_fitted(self):
        if not self.is_fitted:
            raise ValueError('Model is not fitted yet.')
        
    def raw_scores(self):
        self._check_if_fitted()
        chunks = []
        df = self.load_data(self.data_index, self.selected_features)
        for clf in self.clf_replicas:
            # when a gender were missing in the training data
            # (e.g., a gender-specific disease) predict 0 for it
            chunks.append(clf.predict_proba(df, avg=True))
        return pd.concat(chunks, axis=1)
        
    @cached_property
    def ukb_scores(self):
        """Phenotype prediction results for the entire UKB dataset.
        Result includes the following columns:
        
        score: phenotype probability score for each UKB patient, an average
          from all replicas
        replica scores: scores from each replica. 
        """
        scores = self.raw_scores()\
            .rename(columns=lambda n: f'replica{n}', copy=False)
        avg_scores = scores.mean(axis=1)
        known = pd.Series(1, self.all_cases)
        return pd.concat(
            [avg_scores, self.full_cohort, known] 
            + [sc for _, sc in scores.items()],
            keys=['score', 'is_case', 'known'] + list(scores),
            axis=1)
    
    @cached_property
    def predicted_cohorts(self):
        """A range of cohorts constructed with different methodologies. 
        """
        cohorts = {}
        scores = (
            self.ukb_scores['score']
            # extend scores to the full UKB index to accomodate all know cases      
            .reindex(self.ukb_genders.index, fill_value=0.0))
        max_ncr = self._conf.analysis.max_ncr
        prevalence = self._conf.analysis.disease_prevalence
        n_cases = len(self.all_cases)
        ukb_size = len(self.ukb_genders)
        if prevalence:
            ukb_prevalence = n_cases / ukb_size
            if ukb_prevalence / prevalence < .99:
                # convert the expected prevalence to NCR
                prevalence_cases = int(prevalence * ukb_size)
                new_ncr = min(max_ncr, prevalence_cases / n_cases)
                logging.info(
                    'Applying prevalence info: '
                    f'expected prevalence is {prevalence}, '
                    f'UKB prevalence is {ukb_prevalence} '
                    f'max NCR is set from {max_ncr} to {new_ncr}.')
                max_ncr = new_ncr
        
        min_score = self._conf.analysis.min_prediction_thresh
        if scores.max() < min_score:
            # nothing will be predicted so Milton won't have any results
            raise ValueError(
                f'All scores are under the threshold of {min_score}. '
                'Not producing any cohorts.')
        ncr = running_ncr(scores, self.all_cases, min_score)
        ncr = self.gender_balanced_ncr(ncr[ncr <= max_ncr])
        
        thresholds = ncr[ncr >= 1].quantile(self._conf.analysis.ncr_quantiles)
        for level, thresh in enumerate(thresholds):
            ix = ncr[ncr <= thresh].index
            cohorts[f'L{level}'] = pd.Series(1.0, index=ix)
        return cohorts
    
    def gender_balanced_ncr(self, ncr):
        """Takes a cohort with NCR scores (in which values <= 1 indicate known 
        labels and values > 1 are predictions) and returns a subset this cohort
        that ensures the same gender proportions between known and predicted 
        cases. This method does not use sampling to remove cases but instead
        removals proceed in the order of decreasing NCR values (increasing
        prediction scores - from least confident to most confident).
        """
        df = pd.concat([ncr, self.ukb_genders[ncr.index]], 
                       axis=1, keys=['NCR', Col.GENDER])
        known = df[df['NCR'] <= 1]
        predicted = df[df['NCR'] > 1]
        if predicted.empty:
            raise ValueError(
                'Empty predicted cohort: cannot perform gender-balancing.')
        with set_random_state(self.rnd):
            df_out = poststratify(predicted, by=[Col.GENDER], like_df=known, 
                                same_size=False, 
                                scores=predicted['NCR'])
            return ncr.loc[known.index.union(df_out.index)]

    @cached_property
    def feature_importance(self):
        self._check_if_fitted()
        coeffs = pd.concat([clf.coefficients() for clf in self.clf_replicas],
                           axis=1)
        avg_coeffs = coeffs.mean(axis=1).sort_values(ascending=False)
        return coeffs.reindex(avg_coeffs.index)
    
    @cached_property
    def _embeddings(self):
        self._check_if_fitted()
        X = self.clf_replicas[0].transform(self.data)
        y = self.full_cohort.reindex(X.index)
        conf = self._conf.output
        plots = Proj2dPlots(
            max_scatter_points=conf.max_scatter_points,
            cmap=conf.cmap)
        return plots.fit(X, y)
    
    def show_performance(self):
        """Displays model performance characteristics along with 
        distribution of sample scores on the validation set.
        """
        roc = plot_multi_curve(self.roc_curves, ('TPR', 'FPR'))\
            .opts(title='ROC', aspect=4/3, responsive=True)
        
        pr = plot_multi_curve(self.pr_curves, ('Precission', 'Recall'))\
            .opts(title='Precision-Recall', aspect=4/3, responsive=True)
        
        # ens_scores = plot_ensemble_score_dist(
        #     self.val_scores.mean(axis=1), 
        #     self.val_y)\
        #     .opts(aspect=4/3, responsive=True)
            
        # ukb_scores = plot_ukb_score_dist(
        #     self.ukb_scores.dropna(subset=['is_case'])['score'],
        #     self.ukb_scores.dropna(subset=['low_conf'])['score'])\
        #     .opts(aspect=4/3, responsive=True)
        
        return (
            roc + pr  
            # + ens_scores + ukb_scores
            ).cols(2)
    
    def show_importance(self):
        """Displays relative feature importance estimates obtained
        from fitting a tree ensemble model (XGBoost or random forest)
        """
        return self._show_importance(['General', 'Male', 'Female'])
    
    def show_effect_size(self):
        """Displays coefficients of logistic regression fit to data
        with automatically removed highly-correlated features.
        """
        return self._show_importance(['General-LR', 'Male-LR', 'Female-LR'])
    
    def _show_importance(self, models):
        selected = [n for n in models if n in self._models]
        if not selected:
            raise ValueError(f'None of the requested models was fit: {models}')
        coeffs = [self._models[name].coefficients() for name in selected]
        return plot_feature_importance(coeffs, selected)
    
    def show_feature_embedding(self, features=None):
        """Displays 2D PCA embedding of either top-n most important
        features or of a list of features specified by name.
        
        Parameters
        ----------
        features : int, or list of strings or None - specification of what 
            to show. When int, shows plots for n most important features,
            When list, shows plots for features selected by name. When None,
            generic PCA plots are shown.
        """
        if features is None:
            p0 = self._embeddings.plot_scree()
            p1 = self._embeddings.plot_projections()
            return p0 + p1
        else:
            if isinstance(features, int):
                names = self.feature_importance.index[:features].to_list()
            elif isinstance(features, list):
                names = features
            else:
                raise ValueError('Unexpected type of feature spec: %s'
                                 % type(features))
            return self._embeddings.plot_projections(names).cols(3)
    
    def save_report(self, 
                    location=None, 
                    model_performance=None,
                    feature_importance=None,
                    effect_sizes=None,
                    embedding=None,
                    feature_embeddings=None,
                    qv_significance=True,
                    **unused):
        if location is None:
            location = self._conf.output.location
        location = Path(location)
            
        if model_performance is None:
            model_performance = self._conf.output.model_performance
            
        if feature_importance is None:
            feature_importance = self._conf.output.feature_importance
            
        if effect_sizes is None:
            effect_sizes = self._conf.output.effect_sizes
            
        if embedding is None:
            embedding = self._conf.output.embedding
            
        if feature_embeddings is None:
            feature_embeddings = self._conf.output.feature_embeddings
        
        plots = {}
            
        if model_performance:
            plots['model_performance'] = self.show_performance()
            
        if feature_importance:
            plots['feature_importance'] = self.show_importance()
            
        if effect_sizes:
            plots['effect_sizes'] = self.show_effect_size()
            
        if embedding:
            plots['embedding'] = self.show_feature_embedding()
            
        if feature_embeddings:
            features = feature_embeddings
            plots['feature_embeddings'] = self.show_feature_embedding(features)
            
        # ensure output directory
        location.mkdir(exist_ok=True)

        self._store_html_report(location, plots)
        self._store_metrics(location)
        self._store_stuff(location)
        self._store_model_coeffs(location)
        self._store_scores(location)
        
        with set_random_state(self.rnd):
            if qv_significance and self.predicted_cohorts:
                self._store_qv_significance(location)
        
    def _store_html_report(self, location, plots):
        bokeh_charts = {k:  hv_render_html(v) 
                        for k, v in plots.items()}
        js_scripts, js_charts = html_components(bokeh_charts)
        css = self._templates.get_template('styles.jinja2').render()
        
        with (location / 'report.html').open('w') as f:
            template = self._templates.get_template('milton-report.jinja2')
            f.write(template.render(
                bokeh_version=bokeh_version,
                title=self._conf.output.title,
                figure_scripts=js_scripts,
                overview=self._repr_html_(),
                settings=str(self.settings),
                css=css,
                **js_charts))

    def _store_metrics(self, location):
        pd.concat([
            self.replica_metrics,
            self.data_metrics
        ], axis=1).to_csv(location / 'metrics.csv', index=False)
        
    def _store_stuff(self, location):
        data = {
            'roc': self.roc_curves,
            'precision-recall': self.pr_curves,
            'hyper-params': self.best_hyper_params,
            'estimator': self.estimator_type
        }
        with (location / 'stuff.pickle').open('wb') as f: 
            pickle.dump(data, f)
        
    def _store_model_coeffs(self, location):
        coefs = pd.concat(
            [m.coefficients() for m in self._models.values()],
            keys=self._models.keys(),
            axis=1)
        replica_ft_imp = self.feature_importance\
            .rename(columns=lambda c: f'replica{c}', copy=False)
        all_coefs = pd.concat([coefs, replica_ft_imp], axis=1)

        # store coefficients for all trained models
        all_coefs.rename_axis('Feature')\
            .to_csv(location / 'model_coeffs.csv')
        
    def _store_qv_significance(self, location):
        cohorts = self.predicted_cohorts.values()
        pred_cohort_names = list(self.predicted_cohorts)
        predicted = [self.qv_significance(cohort) for cohort in cohorts]
        known_cohort = pd.Series(1, self.all_cases)
        result = pd.concat([self.qv_significance(known_cohort)] + predicted,
                           keys=['known'] + pred_cohort_names, 
                           names=['cohort'])
        result.to_parquet(location / 'qv_significance.parquet')
        
    def _store_scores(self, location):
        if self.predicted_cohorts:
            cohorts = pd.DataFrame(self.predicted_cohorts)
            cohorts.columns = [f'cohort_{n}' for n in self.predicted_cohorts]
            data = pd.concat([self.ukb_scores, cohorts], axis=1)
        else:
            data = self.ukb_scores
        data.to_parquet(
            location / 'scores.parquet', 
            engine='pyarrow')
