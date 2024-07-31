from functools import cached_property
import logging
import numpy as np
import pandas as pd
from itertools import product
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
from dask_ml.model_selection import GridSearchCV, RandomizedSearchCV
from .processing import DataTransformer, resample
from .random import RND, randint


class MiltonException(Exception):
    pass


@make_scorer
def specificity(y, y_pred):
    """Specificity score: P(y_pred = y | y = 0)
    """
    return (y_pred[y == 0] == 0).mean()


def stratified_rnd_split(y, test_frac=.15):
    """Splits class vector y into two train/test components
    in provided proportion. Class proportions are preserved, 
    element indices are returned.
    """
    train = []
    test = []
    classes = np.unique(y)
    
    for cls in classes:
        ix = np.flatnonzero(y == cls)
        RND().shuffle(ix)
        
        n = int(len(ix) * (1 - test_frac))
        train.append(ix[:n])
        test.append(ix[n:])
        
    return np.concatenate(train), np.concatenate(test)


def array_splits(array, block_size):
    n = len(array)
    n_chunks = round(n / block_size)
    chunk_size = n / n_chunks
    for i in range(n_chunks):
        yield array[round(i * chunk_size) : round((i + 1) * chunk_size)]


def balanced_partitions(y, shuffle=True):
    """Returns a sequence of balanced partition pairs. Each pair
    comprises case and control indices which are randomly sampled
    from y (no replacement) and of approximately the same size, 
    (which is the size of the smaller of the classes).
    
    The number of pairs produced depends on the degree of imbalance
    between both classes in y.
    
    For example, if class 0 is 2x larger than class 1, expect two
    pairs returned, with the entire class 1 being returned in each
    pair.
    """
    case_ix = np.flatnonzero(y)
    ctl_ix = np.flatnonzero(1 - y)
    
    if shuffle:
        RND().shuffle(case_ix)
        RND().shuffle(ctl_ix)

    case_n = len(case_ix)
    ctl_n = len(ctl_ix)
    
    if case_n < ctl_n:
        yield from product([case_ix], array_splits(ctl_ix, case_n))
    else:
        yield from product(array_splits(case_ix, ctl_n), [ctl_ix])
    

class MiltonPipeline(BaseEstimator, ClassifierMixin):
    """Main pipeline class that fuses together a transformer and the
    ensemble classifier. 
    The class can also run hyper parameter tuning prior to training 
    the pipeline. See documentation of __init__().
    """
    
    def __init__(self, spec, preproc_params=None, **hpt_args):
        """New Milton pipeline instance. 
        
        Parameters
        ----------
        spec : either a sklearn classifier instance, or MultiClfOptimizer
            specification (see the initializer) or None, in which case
            the best-effort (hand-picked) classifier is used.
        preproc_params : a dictionary of parameters to pass to DataTransformer
            or None for the default values.
        **hpt_args : extra keyword arguments to be passed to MultiClfOptimizer
        """
        assert isinstance(spec, (BaseEstimator, list, type(None)))
        self.preproc_params = preproc_params or {}
        self.spec = spec
        self.hpt_args = hpt_args
        self._preproc = None
        self._schema = None
        self._ensemble = []
        
    def fit(self, X, y, **fit_kwargs):
        """Trains the pipeline. Returns self.        
        """
        self._preproc = DataTransformer(**self.preproc_params)
        Xpp = self._preproc.fit_transform(X, y)
        self._schema = Xpp.columns
        self.estimator_ = self._get_estimator(Xpp, y)
        self.partitions_ = [np.hstack(p) for p in balanced_partitions(y)]
        for ix in self.partitions_:
            Xi = Xpp.iloc[ix]
            est = clone(self.estimator_).fit(Xi, y.iloc[ix], **fit_kwargs)
            self._ensemble.append(est)
        self.classes_ = np.unique(y)
        return self
            
    def _get_estimator(self, X, y):
        if isinstance(self.spec, BaseEstimator):
            return self.spec
        elif self.spec is None:
            return XGBClassifier(random_state=randint(), **self.BEST_XGB_PARAMS)
        else:
            hpt = MultiClfOptimizer(spec=self.spec, **self.hpt_args)
            X_rs = resample(X, y, same_size=True)
            hpt.fit(X_rs, y.reindex(X_rs.index))
            return hpt.best_clf_
        
    def _ens_predict(self, X):
        pred_lst = [est.predict(X) for est in self._ensemble]
        avg_pred = np.vstack(pred_lst).mean(axis=0)
        return pd.Series((avg_pred >= .5).astype('int'), X.index)
    
    def _ens_predict_proba(self, X, avg=True):
        pred_lst = [est.predict_proba(X)[:, 1] for est in self._ensemble]
        probs = np.vstack(pred_lst).T
        if avg:
            result = pd.Series(probs.mean(axis=1), X.index)
        else:
            result = pd.DataFrame(probs, X.index)
        return  result
    
    @property
    def feature_names(self):
        return self._schema
    
    def transform(self, X):
        """Transforms X with the preprocessor. Requires fitted pipeline.
        """
        return self._preproc.transform(X)
    
    def predict(self, X):
        """Predicts on X. Requires fitted pipeline. 
        """
        X = self._preproc.transform(X)
        return self._ens_predict(X)
    
    def predict_proba(self, X, avg=True):
        """Computes classification probability score. Requires fitted 
        pipeline. Synchronized operation.
        
        Parameters
        ----------
        X : input DataFrame
        avg : When False, returns scores from individual ensemble
            members, otherwise averaged scores are produced.
        """
        X = self._preproc.transform(X)
        return self._ens_predict_proba(X, avg)
    
    # required by sklearn for cross-validation
    def decision_function(self, X):
        return self.predict_proba(X, avg=True).values.reshape((-1, 1))
    
    def coefficients(self, avg=True):
        """Computes model coefficients. Regardless of the type of
        estimator used (logistic regression or tree models)
        
        Parameters
        ----------
        avg : When False, returns coefficients from individual ensemble
            members, otherwise averaged coefficients are produced.
        """
        def get_coefs(est):
            if isinstance(est, LogisticRegression):
                return est.coef_[0]
            elif isinstance(est, (
                DecisionTreeClassifier, 
                RandomForestClassifier, 
                XGBClassifier)):
                return est.feature_importances_
            else:
                # unsupported models
                return pd.Series(
                    [], 
                    dtype='float64', 
                    index=pd.Index([], dtype='str'))
        coefs = [get_coefs(est) for est in self._ensemble]
        names = self.feature_names
        df = pd.concat([pd.Series(c, index=names) for c in coefs], axis=1)
        if avg:
            return df.mean(axis=1).sort_values(ascending=False)
        else:
            return df
    

class MultiClfOptimizer:
    """Multi-Classifier Hyper Parameter Tuning. A wrapper for scikit's 
    GridSearchCV for running HPT on more than one classifier.
    
    The optimizer uses Dasks's GridSearchCV as computational backend.
    """
    
    def __init__(self, 
                 spec, 
                 refit=False,
                 metrics=('f1', 'roc_auc', 'precision', 'recall'), 
                 selection_metric='f1',
                 cv=5,
                 grid_search=30):
        """Creates a new a new optimizer instance. 
        
        Parameters
        ----------
        spec : a sequence of (classifier, param grid) pairs, or
            (name, classifier, param_grid) tripples. 
        refit : If True (default), the best classifier is refit to 
            the training set.
        metrics : list of sklearn's metrics to evaluate
        cv : number of CV splits
        grid_search : if True, GridSearchCV is used, otherwise the
            parameter must be an integer and denotes the number of 
            iteration in RandomizedSearchCV.
        """
        assert selection_metric in metrics
        assert isinstance(grid_search, (bool, int))
        
        self.names = []
        self.classifiers = []
        self.param_grids = []
        self.refit = refit
        self.metrics = list(metrics)
        self.selection_metric = selection_metric
        self.cv = cv
        self.grid_search = grid_search
        self._parse_spec(spec)
        
    def _parse_spec(self, spec):
        assert len(spec), 'Expecting at least one specification.'
        for s in spec:
            assert len(s) in (2, 3)
            if len(s) == 2:
                clf, param_grid = s
                name = clf.__class__.__name__
            else:
                name, clf, param_grid = s
            self.names.append(name)
            self.classifiers.append(clf)
            self.param_grids.append(param_grid)
        
    def fit(self, X, y):
        results = []
        exception = None
        X, y = check_X_y(X, y)
        
        if self.grid_search == True:
            optimizer = GridSearchCV  
        else:
            def optimizer(*args, **kwargs):
                return RandomizedSearchCV(*args, n_iter=self.grid_search, 
                                          **kwargs)
        for i in range(len(self.classifiers)):
            try:
                param_grid = self.param_grids[i]
                # use grid search for small configurations
                n = sum(1 for e in product(*param_grid.values()))
                opt = optimizer if n >= 20 else GridSearchCV
                hpt = opt(
                    self.classifiers[i], 
                    param_grid, 
                    scoring=self.metrics, 
                    refit=False, 
                    cv=self.cv, 
                    n_jobs=-1)
                hpt.fit(X, y)
                results.append(hpt.cv_results_)
            except Exception as ex:
                logging.error(ex)
                exception = ex
       
        # best-effort error handling
        if exception:
            if not results:
                raise Exception('Hyper Parameter Tuning error') from exception
            else:
                logging.exception(exception)
        
        self._proc_results(results)
        if self.refit:
            self.best_clf_.fit(X, y)
            
    def _proc_results(self, results):
        clf_best_metrics = []
        clf_best_ix = []
        selector = 'mean_test_' + self.selection_metric
        metric_keys = ['mean_test_' + m for m in self.metrics]
        all_results = []

        for _, res in zip(self.names, results):
            mvals = res[selector]
            clf_best_metrics.append(mvals.max())
            clf_best_ix.append(np.argmax(mvals))
            all_metrics = {mk: res[mk].max() for mk in metric_keys}
            all_results.append(all_metrics)

        self.results_ = pd.DataFrame(all_results, index=self.names)
        best = np.argmax(clf_best_metrics)
        self.best_params_ = results[best]['params'][clf_best_ix[best]]
        self.best_clf_ = self.classifiers[best]
        self.best_clf_.set_params(**self.best_params_)
        self.raw_results_ = results
