"""UKB Patient/Subject Selector - a high level interface cohort building.
"""
from functools import wraps
import numpy as np
import pandas as pd
from dask import delayed
from typing import Optional
from contextlib import contextmanager

from .icd10 import ICD10Tree
from .random import RND
from .data_desc import Col
from .data_source import UkbDataStore
from .data_info import (
    UKB,
    HESIN_DIAG,
    DEATH_CAUSE)
from .utils import (
    find_values, 
    unique_boolean_agg,
    stratified_sample)


def h_concat(parts, *args, **kwargs):
    """Delayed horizontal concatenation of pandas objects. 
    """
    @wraps(pd.concat)
    def concat(parts):
        return pd.concat(parts, *args, axis=1, copy=False, **kwargs)
    
    return delayed(concat)(parts)


class UkbPatientSelector:
    # generic UKB contol group will be N-times larger than the cases
    UKB_CONTROLS_CASES_RATIO = 4  
    
    # currently supported stratification factors
    SAMPLING_FACTORS = [Col.GENDER, Col.AGE]
    
    def __init__(self, 
                 *,
                 data_store=None,
                 data_dict=None,
                 location=None,
                 controls=None, 
                 diag_fields=(41270,),
                 use_icd10_extras=True,
                 ukb_controls_cases_ratio=UKB_CONTROLS_CASES_RATIO,
                 infrequent_gender_thresh=0.0,
                 sampling_factors=None,
                 cached=True):
        """Creates a patient selector.
        
        Parameters
        ----------
        data_store : the data store to use. 
        data_dict : optional UKB data dictionary instance
        location : optional location to provide to the default data store 
          initializer when data_store is not provided
        controls : None, sequence of integers
          When None, sample the controls from the entire UKB population, 
          otherwise expect concrete subject IDs
        ukb_controls_cases_ratio : float,
          A generic UKB contol group will be N-times larger than the cases.
        infrequent_gender_thresh : float, 0 <= x < .5
          If the less frequent gender has a frequency under this thresh it is
          excluded from the results
        diag_fields : Sequence of integers
          UKB fields to fetch diagnoses from
        use_icd10_extras : boolean
          Whether to search additional datasets (HES) for ICD10 diagnoses
        sampling_factors : optional seq of pairs of (feature name, num bins)
          to define which factors should be considered in sampling the controls
          in order to match data distribution in cases. The num bins values
          specify the number of of bins to use when discretizing a factor. 
          Use 0 when raw values should be used. 
          WARNING: currently, the factors must be either [age, gender] or None
        cached : whether to cache underlying data in memory for faster
            evaluations (makes sense for more than one invocation).
        """
        self.dst = data_store or UkbDataStore(location=location, cached=cached)
        self.location = self.dst.location
        self.dd = data_dict or self.dst.ukb.data_dict
        self.ukb_controls_cases_ratio = (ukb_controls_cases_ratio
                                         or self.UKB_CONTROLS_CASES_RATIO)
        if infrequent_gender_thresh < .5 and infrequent_gender_thresh >= 0.0:
            self.infrequent_gender_thresh = infrequent_gender_thresh
        else:
            raise ValueError(
                'Infrequent gender threshold must be a float in [0, .5)')
        self.diag_fields = diag_fields
        self.use_icd10_extras = use_icd10_extras
        if sampling_factors:
            fact_names = [n for n, _ in sampling_factors]
            if Col.GENDER not in fact_names:
                raise ValueError(f'"{Col.GENDER}" is a mandatory sampling factor')
            if not set(fact_names).issubset(self.SAMPLING_FACTORS):
                raise ValueError('Supported sampling factors are: ',
                                 f'{self.SAMPLING_FACTORS}')
            self.sampling_factors = self._load_factors(dict(sampling_factors))
            self.ukb_subjects = self.sampling_factors.index
        else:
            self.sampling_factors = None
            # read all UKB subject IDs
            self.ukb_subjects = self.dst[UKB].dataset.index
        if controls is None:
            self.controls = self.ukb_subjects
        else:
            self.controls = pd.Index(controls).intersection(self.ukb_subjects)
                
    @property
    def cached(self):
        return self.dst.cached
                
    @property
    def opt_outs(self):
        """pd.Index of UKB subject IDs that are excluded from all processing
        (or None if not defined).
        Note: the underlying datasource removes those IDs from its results 
              anyway and would not be returned by any of the evaluation methods.
        """
        return self.dst.opt_outs

    def _load_factors(self, factors):
        fields = self.dd.find(title=sorted(factors.keys()))
        if len(fields) != len(factors):
            raise ValueError(f'Ambiguous field names: {sorted(factors)}')
        df = self.dst.ukb.read_processed(self.dd.to_schema(fields))
        
        for fname, nbins in factors.items():
            if nbins:
                bins = df[fname].quantile(np.linspace(0, 1, nbins + 1))
                df[fname] = pd.cut(df[fname], bins, include_lowest=True)
        return df.dropna().sort_index() 
            
    @staticmethod
    def _agg_result(bool_series):
        """Ensures the boolean series has unique index. Multiple
        records for the same patient ids are grouped together via
        logical OR operation.
        """
        return unique_boolean_agg(bool_series)
    
    def _add_controls(self, 
                      cases: pd.Index, 
                      ctl_exclusions: Optional[pd.Index] = None):
        controls = self.controls.difference(cases)
        if ctl_exclusions is not None:
            controls = controls.difference(ctl_exclusions)
        if self.sampling_factors is None:
            # simple sub-sampling of controls to achieve the desired ratio
            n = min(len(cases) * self.ukb_controls_cases_ratio, len(controls))
            controls = pd.Index(RND().choice(controls, n, replace=False))
            return pd.concat([
                pd.Series(True, cases, name='is_case'), 
                pd.Series(False, controls, name='is_case')])
        else:
            return self._stratified_sample(cases, controls)
    
    def _stratified_sample(self, cases: pd.Index, controls: pd.Index):
        result = stratified_sample(
            self.sampling_factors.loc[controls].assign(is_case=False),  
            self.sampling_factors.loc[cases].assign(is_case=True),
            self.sampling_factors.columns.to_list(),
            size_factor=self.ukb_controls_cases_ratio)
        # drop the sampling factors
        return result['is_case']  
    
    def eval_icd10(self, 
                   codes, 
                   ctl_exclusions=None,
                   ukb_fields=None,
                   raw=False):
        """Produces case/control patient IDs for the ICD10 codes provided. 
        Searches for the codes in the UKB, hesin_diag and death_cause data sets.
        
        Parameters
        ----------
        icd10_codes : list of str,
          Collection of ICD10 (strings) to search for.
        ctl_exclusions : optional list of str
          ICD10 codes to be excluded from control group
        ukb_fields : Sequence of int, optional
          UKB field IDs to seach in. If not provided, use the defaults 
        raw : bool,
          Returns all partial diagnoses as multiple columns in a data frame
        
        Returns
        -------
        pd.Index with UKB subject IDs containing diagnoses in icd10_codes and
        (when ctl_exclusions was provided) another pd.Index with subject IDs
        with diagnoses in the set of excluded codes.
        """
        codes = set(codes)
        ctl_exclusions = set(ctl_exclusions or set()).difference(codes)
        fields = self.dd.find(field_id=list(ukb_fields or self.diag_fields))
        schema = self.dd.to_schema(fields)
        
        def find_codes(df, name):
            cases = find_values(df, codes)\
                .rename(name, copy=False)
            if ctl_exclusions:
                controls = find_values(df, ctl_exclusions)\
                    .rename(name + '_excl', copy=False)
                return pd.concat([cases, controls], axis=1)
            else:
                return cases.to_frame()
                        
        components = ['ukb']
        ukb_matches = self.dst.ukb.read_data(schema)\
            .map(find_codes, 'ukb')\
            .concat()
        
        if self.use_icd10_extras:
            hesin = self.dst[HESIN_DIAG].dataset\
                .load(['diag_icd10'])\
                .map(find_codes, 'hesin')\
                .map(self._agg_result)\
                .concat()
            
            death = self.dst[DEATH_CAUSE].dataset\
                .load(['cause_icd10'])\
                .map(find_codes, 'death')\
                .map(self._agg_result)\
                .concat()
            
            res = h_concat([
                ukb_matches,
                delayed(self._agg_result)(hesin),
                delayed(self._agg_result)(death)
            ]).compute().fillna(False)
            components.extend(['hesin', 'death'])
        else:
            res = ukb_matches.compute()
        
        res.sort_index(inplace=True)  
        
        if not raw:
            expr = ' | '.join(components)
            cases = res.eval(expr)\
                .loc[lambda s: s == True]\
                .index
            if ctl_exclusions:
                expr = ' | '.join([c + '_excl' for c in components])
                non_controls = res.eval(expr)\
                    .loc[lambda s: s == True]\
                    .index
                return cases, non_controls
            return cases
        else:
            return res
    
    def __call__(self, cohort_spec, 
                 add_controls=True, 
                 ukb_controls=False, 
                 case_exclusions=None,
                 ctl_exclusions=None,
                 drop_same_chapter_ctl=True,
                 fraction=None,
                 ret_known_cases=False):
        """Generic patient evaluation. 
        
        Parameters
        ----------
        cohort_spec : ICD10Tree, list/tuple of str ICD10 codes, pd.Index
          Definition of cases: tu be extended with suitable control set.
        add_controls : bool or int,
          Whether to add controls to the result, when False, the result contains
          only the cases, when positive integer, returns this many cohorts with
          different random samplings of controls
        ukb_controls : boolean
          When True, the control settings specified in the constructor are 
          ignored and the all UKB non-cases are used as controls
        case_exclusions : sequence of int IDs, optional
          Subject IDs to exclude from the matched cases. This is useful when
          you want a subset of cases to be excluded *before* controls are added,
          i.e., to have control distribution match the cases minus exclusions.
        drop_same_chapter_ctl : boolean
          When the specification is an ICD10Tree object, drops controls from
          the same chapter as the respective ICD10 code.
        ctl_exclusions : sqeuence of int IDs, optional
          Subject IDs to exclude from controls
        fraction : float in (0, 1), optional,
          Random fraction of matching cases to use.
        ret_known_cases : bool, optional
          Whether to return the full set of cases matching the specification,
          so not subject to exclusions and resampling.
        """
        if ctl_exclusions is not None:
            ctl_exclusions = pd.Index(ctl_exclusions)
        else:
            ctl_exclusions = pd.Index([], dtype='int')
        if isinstance(cohort_spec, ICD10Tree):
            exclusions = set()
            if drop_same_chapter_ctl:
                # control group should exclude all codes from the same chapter
                exclusions = set(cohort_spec.chapter).difference(set(cohort_spec))
                matched_ids, excluded_ids = self.eval_icd10(cohort_spec, exclusions)
                ctl_exclusions = ctl_exclusions.union(excluded_ids)
            else:
                matched_ids = self.eval_icd10(cohort_spec)
        elif isinstance(cohort_spec, (list, tuple)):
            matched_ids = self.eval_icd10(cohort_spec)
        elif isinstance(cohort_spec, pd.Index):
            matched_ids = cohort_spec.unique()
        else:  
            raise ValueError(
                'Illegal type of patient selection method: '
                + str(type(cohort_spec)))
            
        known_cases = matched_ids 
        if case_exclusions is not None:
            matched_ids = matched_ids.difference(case_exclusions)
        
        if self.infrequent_gender_thresh:
            # remove gender whose frequency is under the threshold
            genders = self.sampling_factors.loc[matched_ids, Col.GENDER]
            freq = genders.value_counts(normalize=True)
            if freq.min() < self.infrequent_gender_thresh:
                discarded_gender = freq.index[freq.argmin()]
                matched_ids = genders.index[genders != discarded_gender]
        
        if fraction is not None:
            if fraction <= 0 or fraction >= 1:
                raise ValueError('fraction must be a float in (0, 1).')
            n = int(fraction * len(matched_ids))
            matched_ids = np.random.choice(matched_ids, size=n, replace=False)
        
        cohort = pd.Series(True, index=matched_ids, name='is_case')
        if add_controls:
            if ukb_controls:
                result = cohort.reindex(self.ukb_subjects, fill_value=False)
            elif add_controls is True:
                result = self._add_controls(matched_ids, ctl_exclusions)
            else:
                result = [self._add_controls(matched_ids, ctl_exclusions)
                        for _ in range(int(add_controls))]
        else:
            result = cohort
            
        return (result, known_cases) if ret_known_cases else result

    @contextmanager
    def options(self, **opts):
        old_vals = {
            name:getattr(self, name) for name in opts
        }
        try:
            for name, value in opts.items():
                setattr(self, name, value)
            yield self
        finally:
            for name, value in old_vals.items():
                setattr(self, name, value)
