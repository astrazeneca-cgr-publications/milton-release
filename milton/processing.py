import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Union
from numbers import Number
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler

from .data_desc import Col
from .random import RND


def poststratify(df, by, like_df, same_size=False, scores=None):
    """ Resamples df stratifying by given columns (by) so that
    the resulting relative group sizes are like in like_df.
    
    Parameters
    ----------
    df : the data frame to resample
    by : list of columns to stratify by, or a single column name
    like_df : reference data frame with expected group sizes
    same_size : if true, like_df is expected to have <= rows in
        in each group than df so that the result can be sampled
        down to match like_df in length.
    """
    src_groupby = df.groupby(by)
    src_counts = src_groupby.size()
    src_freq = src_counts / len(df)
    
    dst_counts = like_df.groupby(by).size()
    dst_freq = dst_counts / len(like_df)
    weights = dst_freq / src_freq
    
    if same_size:
        grp_sizes = dst_counts
    else:
        grp_sizes = (weights * (1 / weights).min() * src_counts)\
            .dropna()\
            .astype('int')\
            .to_dict()
            
    def select_samples(ix, n):
        if scores is None:
            return RND().choice(ix, n, replace=False)
        else:
            grp_scores = scores.loc[ix].sort_values()
            return grp_scores.index[:n]
    
    # stratified sampling: sample from every group
    ix_lst = []
    for group, ix in src_groupby.groups.items():
        if group in grp_sizes:
            n = grp_sizes[group]
            ix_lst.append(select_samples(ix, n))
            
    return df.loc[np.hstack(ix_lst)]


def resample(df, classes, stratify_by=None, same_size=True):
    """ Resampling of df when split into binary classes. Optional class
    stratification by a number of factors (columns in df) can be performed.
    
    Parameters
    ----------
    df : dataframe to resample
    classes : binary series defining the two classes, identically indexed as df
    stratify_by : list of factors to stratify both classes by (larger class will
        be re-sampled to match the group frequencies of the smaller one)
    same_size : whether the result should have equal class proportions
    """
    classes = classes.reindex_like(df)
    small_class = int(classes.mean() <= .5)
    smaller = df[classes == small_class]
    larger = df[classes == 1 - small_class]
    
    if stratify_by is None:
        if same_size:
            larger = larger.sample(len(smaller), random_state=RND())
    else:
        larger = poststratify(larger, by=stratify_by, 
                              like_df=smaller, 
                              same_size=same_size)
        
    return pd.concat([smaller, larger])


def clip_quantiles(series, low_q=0, high_q=1):
    """ Caps all values outside given quantile range.
    """
    q0 = series.quantile(low_q)
    q1 = series.quantile(high_q)
    return series.mask(series < q0, q0).mask(series > q1, q1)


def trim_quantiles(series, low_q=0, high_q=1):
    """ Removes all values that fall outside the given
    quantile range.
    """
    low, high = series.quantile([low_q, high_q])
    return series[(series >= low) & (series <= high)]


class GenderSpecNAStrategy:
    """Gender-specific strategy of NA imputation.
    """
    def __init__(self, 
                 males: Union[float, str] = 'median',
                 females: Union[float, str] = 'median'):
        self.males = males
        self.females = females
        
    def __eq__(self, other):
        return self.males == other.males and self.females == other.females
    
    def __hash__(self) -> int:
        return hash((self.males, self.females))
    
    def __str__(self) -> str:
        return (
            f'{self.__class__.__name__}[males: {self.males}, '
            f'females: {self.females}]')


class CategoricalNAStrategy:
    """Basic strategy of adding a dedicated column for NA values in one-hot
    encoding of categorical values.
    """
    pass
    
    
class GenderSpecImputer:
    """Gender-specific NA imputer.
    """
    def __init__(self, strategy: GenderSpecNAStrategy):
        self.strategy = strategy
        
    def fit(self, df: pd.DataFrame, genders: pd.Series):
        fill_vals = {}
        males = df[genders == 'Male']
        females = df[genders == 'Female']
        for col in df.columns:
            fill_vals[col] = (
                self._fill_value(males[col], self.strategy.males),
                self._fill_value(females[col], self.strategy.females)
            )
        self.fill_vals_ = fill_vals
        return self
        
    def transform(self, df: pd.DataFrame, genders: pd.Series):
        result = df.copy()
        male_msk = genders == 'Male'
        female_msk = genders == 'Female'
        for name, (male_val, female_val) in self.fill_vals_.items():
            result[name] = (
                df[name]
                .mask(male_msk, male_val)
                .mask(female_msk, female_val))
        # return a numpy array for consistency with sklearn
        return result.to_numpy()
        
    def _fill_value(self, s: pd.Series, value: Union[str, float]):
        if value in ('mean', 'median'):
            # compute the standard functions: mean/median
            fill_val = getattr(s, value)()
        elif isinstance(value, Number):
            fill_val = float(value)
        else:
            raise ValueError(f'Unrecognized imputation value: {value}')
        return 0.0 if np.isnan(fill_val) else fill_val


class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 drop_cols=None,
                 dont_drop=None,
                 drop_na_frac=1.0,
                 drop_correlated=False,
                 na_imputation='median',
                 na_imputation_extra=None,
                 scaling='standard',
                 encode_bin_na=False):
        """Input data transformer that performs a range of parameterized
        and standard data transformations, requires fitting.
        
        Parameters
        ----------
        drop_cols : list of column names to explicitly drop
        dont_drop : a list of columns that *should not* be dropped 
        drop_na_frac : drop columns with the given minimim fraction of N/As.
        drop_correlated : when float in (0, 1), it is a minimum correlation
          coefficient value to drop a column (one of the correlated pair).
          When False, do nothing.
        na_imputation : str, optional
          StandardImputer strategy to use: 'mean', 'median', 'most_frequent', 
          None, ('constant', value) or GenderSpecNAStrategy to calculate 
          replacement for missing values. 
        na_imputation_extra: dict, optional
          NA imputation settings for specific columns, in addition to the 
          default setting. Keys are column names.
        scaling : one of ('power', 'standard', None). Applies PowerTransformer, 
          StandardScaler, or no scaling at all.
        encode_bin_na : Wether NAs in binary categorical variables should be 
          one hot-encoded as a vector of zeros. Otherwise, ignore NAs and use 
          default encoding logic.
        """
        self.drop_cols = drop_cols
        self.dont_drop = dont_drop
        self.drop_na_frac = drop_na_frac
        self.drop_correlated = drop_correlated
        self.encode_bin_na = encode_bin_na
        self.na_imputation_extra = na_imputation_extra or {}
        self.cat_na_imputation = set()  # categorical features
        self.na_imputation = na_imputation
        if scaling in ['power', 'standard', None]:
            self.scaling = scaling
        else:
            raise ValueError(
                'Unrecognized value of scaling parameter: {scaling}')
        
    def _initialize(self):
        self.cols_to_drop_ = set()
        self.na_fractions_ = None
        self.num_cols_ = None
        
        if self.scaling == 'power':
            self._rescaler = PowerTransformer() 
        elif self.scaling == 'standard':
            self._rescaler = StandardScaler()
        else:
            self._rescaler = None
    
    def _corr_to_drop(self, df, method='spearman'):
        thresh = self.drop_correlated
        corr = df.corr(method).abs()
        to_drop = corr.where(np.triu(corr > thresh, 1)).min().dropna()
        return to_drop.index.to_list()
            
    def fit(self, df, y=None):
        self._initialize()
        gender = df[Col.GENDER] if Col.GENDER in df else None
        if self.drop_cols:
            self.cols_to_drop_.update(self.drop_cols)
            
        self.num_cols_ = df.select_dtypes(include=[np.number]).columns
        self.cat_cols_ = df.columns[df.dtypes == 'category']
        
        # drops columns with too many NAs 
        na_fractions = df[self.num_cols_]\
            .pipe(lambda df: ((df == 0) | df.isna()).mean())\
            .loc[lambda s: s >= self.drop_na_frac]
            
        if not na_fractions.empty:
            self.na_fractions_ = na_fractions 
            self.cols_to_drop_.update(na_fractions.index.to_list())
            
        if self.drop_correlated:
            self.cols_to_drop_.update(self._corr_to_drop(df))
            
        self.cols_to_drop_ -= set(self.dont_drop or [])
            
        if self.cols_to_drop_:
            self.num_cols_ = self.num_cols_.difference(self.cols_to_drop_)
            df = df[df.columns.difference(self.cols_to_drop_)]
        
        imputable_cols = self.num_cols_.union(self.cat_cols_)
        impute_na = self.na_imputation or self.na_imputation_extra
        if not imputable_cols.empty and impute_na:
            self.imputers_ = self._fit_num_imputers(df[imputable_cols], gender)
            
        if not self.num_cols_.empty and self._rescaler:
                x = df[self.num_cols_]
                if self.scaling == 'power':
                    # Due to https://github.com/scikit-learn/scikit-learn/issues/14959
                    # power transform is numerically unstable
                    # scaling data down by a constant factor seems to fix the 
                    # problem (and doesn't affect the result)
                    x = x / 10
                self._rescaler.fit(x)
        return self
    
    def _fit_num_imputers(self, df: pd.DataFrame, gender: pd.Series):
        """Sets up NA imputation for numeric and categorical features.
        Default categorical NA imputation assumes vectors of 0s in 1-hot 
        encoding. Dedicated stratedy (na_imputation_extra) will result in adding
        dedicated nan columns to encoded features.
        """
        if self.na_imputation is None:
            # No NA imputation for all columns
            return {
                tuple(df.column): None
            }
        strat_map = defaultdict(list)
        for col in df.columns:
            # check for feature-specific strategies first
            strategy = self.na_imputation_extra.get(col)
            if strategy is None:
                if col in self.num_cols_:
                    # default strategy for numeric columns
                    strat_map[self.na_imputation].append(col)
            elif isinstance(strategy, CategoricalNAStrategy):
                self.cat_na_imputation.add(col)
            else:
                strat_map[strategy].append(col)
        imputers = {}
        for strategy, columns in strat_map.items():
            if isinstance(strategy, str):
                imp = SimpleImputer(strategy=strategy).fit(df[columns])
            elif isinstance(strategy, GenderSpecNAStrategy):
                imp = GenderSpecImputer(strategy)
                imp.fit(df[columns], gender)
            elif isinstance(strategy, tuple):
                _, value = strategy
                non_num_cols = set(columns).difference(self.num_cols_)
                if isinstance(value, Number) and len(non_num_cols) == 0:
                    imp = SimpleImputer(strategy='constant', fill_value=value)
                    imp.fit(df[columns])
                else:
                    raise ValueError(
                        f'Expecting a number as NA replacement value: {value}')
            imputers[tuple(columns)] = imp
        return imputers
    
    def _impute_na_values(self, df: pd.DataFrame, gender: pd.Series):
        parts = []
        for columns, imputer in self.imputers_.items():
            if imputer is None:
                parts.append(df[list(columns)])
            else:
                x0 = df[list(columns)]
                if isinstance(imputer, GenderSpecImputer):
                    x1 = imputer.transform(x0, gender)
                else:
                    x1 = imputer.transform(x0)
                X = pd.DataFrame(x1, index=df.index, columns=columns)
                parts.append(X)
        return pd.concat(parts, axis=1).loc[:, df.columns]
            
    def transform(self, df):
        check_is_fitted(self)
        gender = df[Col.GENDER] if Col.GENDER in df else None
        df = df[df.columns.difference(self.cols_to_drop_)].copy()
        if not self.num_cols_.empty:
            if self.na_imputation:
                df[self.num_cols_] = self._impute_na_values(
                    df[self.num_cols_], gender)
            if self._rescaler:
                x = df[self.num_cols_]
                if self.scaling == 'power':
                    # for explanation, see comment in .fit()
                    x = x / 10
                df[self.num_cols_] = self._rescaler.transform(x)
        return self._encode_categoricals(df)
    
    def _encode_categoricals(self, df):
        """One-hot encoding of categorical columns: distinguish two-valued 
        categoricals to convert them into a single binary column of 1s/0s when
        there are NAs. Otherwise, perform standard 1-hot encoding adding 
        dedicated NA column where indicated by configuration.
        """
        two_cat = []
        many_cat = []
        cat_cols = df.columns[df.dtypes == 'category']
        for c in cat_cols:
            is_many_cat = len(df[c].cat.categories) > 2
            has_nulls = df[c].isna().sum() > 0
            if c not in self.cat_na_imputation:
                if is_many_cat or has_nulls:
                    many_cat.append(c)
                else:
                    two_cat.append(c)
        if many_cat:
            # Any NAs are represented as vector of 0s
            df = pd.get_dummies(
                df, prefix_sep='::', 
                columns=many_cat, 
                dtype='float')
        if self.cat_na_imputation:
            # adds dedicated NA column 
            df = pd.get_dummies(
                df, prefix_sep='::', 
                columns=list(self.cat_na_imputation), 
                dummy_na=True,
                dtype='float')
        # Standard binary encoding of 2-valued categoricals
        for col in two_cat:
            df[col] = df[col].cat.codes.astype('float') 
        return df
