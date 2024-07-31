import numpy as np
import pandas as pd
from collections import defaultdict
from pandas.api.types import is_numeric_dtype
from pandas.api.types import union_categoricals as _union_categoricals
from pandas.core.frame import DataFrame
from scipy.stats import fisher_exact, chi2_contingency
import numba
from numba.typed import List
from scipy.optimize import minimize_scalar
import re

from .random import RND


def is_monotonic(pandas_obj):
    """Wrapper for the .is_monotonic property in pandas which was deprecated
    in version 1.5. Use to have your code work in both older and newer pandas.
    """
    try:
        return pandas_obj.is_monotonic
    except AttributeError:
        return (pandas_obj.is_monotonic_increasing 
                or pandas_obj.is_monotonic_decreasing)


def concat_indexes(ix_list, *args, **kwargs):
    """Concatenates a list of (non-hierarchical) pd.Index objects.
    The args and kwargs are the same as in pd.concat(objects, *args, **kwargs)
    """
    # reusing pd.concat() to handle all sorts of crazy indexes
    return pd.concat([pd.Series(True, index=ix) for ix in ix_list],
                     *args, **kwargs).index
    
    
def union_categoricals(categoricals):
    """A wrapper for a pandas function with the same that correctly handles 
    the case in which one of the categoricals is all NA (pandas raises
    an exception). For more details, see: union_categoricals() in 
    pandas.api.types
    """
    parts = []
    if len(categoricals):
        dtype_for_empty = None
        for c in categoricals:
            if not c.cat.categories.empty:
                dtype_for_empty = c.dtype
                break
        else:
            # all categoricals are empty - use the first dtype to coerce all
            dtype_for_empty = categoricals[0].dtype
        for c in categoricals:
            if c.cat.categories.empty:
                c = c.astype(dtype_for_empty, copy=False)
            parts.append(c)
    return _union_categoricals(parts)


def v_concat_with_categories(parts, *args, **kwargs):
    """Vertical concatenation (axis=0) of data frames/series that 
    preserves categorical dtypes of columns (As opposed to 
    pd.concat(df_parts), resulting column types of categorical columns 
    in the inputs will always be categorical after concatenation).
    
    Note, the first data frame in the list is used to determine the output
    schema so there must be consistency of column types across parts.
    
    Parameters
    ----------
    parts : list of pd.DataFrame or pd.Series objects (with or without 
      categorical columns)
    args : positional arguments to pd.concat(objs, ...)
    kwargs : keyword arguments to pd.concat(objs, ...)
    """
    if len(parts) == 0:
        return pd.DataFrame()
    else:
        p0 = parts[0]
    
    if isinstance(p0, pd.Series):
        is_dataframe = False
        has_cat_cols = p0.dtype.name == 'category'
    elif isinstance(p0, pd.DataFrame):
        is_dataframe = True
        cat_cols = p0.dtypes == 'category'
        has_cat_cols = cat_cols.any()
    else:
        raise ValueError('Expecting a list of either pd.Series or '
                         'pd.DataFrame objects')
        
    if has_cat_cols:
        common_ix = concat_indexes([p.index for p in parts], *args, **kwargs)
        if not is_dataframe:
            values = union_categoricals(parts)
            return pd.Series(values, index=common_ix, name=p0.name)
        else:
            cat_col_ix = np.flatnonzero(cat_cols)
            cat_df = pd.DataFrame({
                i: union_categoricals([df.iloc[:, i] for df in parts])
                for i in cat_col_ix},
                index=common_ix)
            # this is to enable non-unique column sets
            cat_df.columns = p0.columns[cat_col_ix]
            
            non_cat_cols = ~cat_cols
            if non_cat_cols.any():
                rest_df = pd.concat([df.loc[:, non_cat_cols] for df in parts],
                                    *args, **kwargs)
                return pd.concat([cat_df, rest_df], axis=1)
            else:
                return cat_df
    else:
        return pd.concat(parts, *args, **kwargs)


def all_numeric(df):
    """Returns True if df comprises only numeric columns.
    """
    return df.dtypes.apply(is_numeric_dtype).all()


def all_categorical(df):
    """Returns True if df comprises only categorical columns.
    """
    return (df.dtypes == 'category').all()


def remap_categories(s, cat_map):
    """Renames categories of a categorical series. The function supports
    non-unique category maps (as opposed to pandas), meaning that some 
    categories can be mapped to the same target category.

    Note: This implementation is rather SLOW.
    """
    return pd.Series(pd.Categorical(s.array.map(cat_map, na_action=None)),
                     name=s.name,
                     index=s.index)


@numba.jit(nopython=True)
def _first_non_null_column(arr):
    """Picks the first non-null value column-wise for every row.
    """
    N, M = arr.shape
    result = np.full(N, np.nan)
    
    for i in range(N):
        for j in range(M):
            if not np.isnan(arr[i, j]):
                result[i] = arr[i, j]
                break
        
    return result


@numba.jit(nopython=True)
def _first_non_null_cat_column(arr_lst):
    """Picks the first non-null value column-wise for every row.
    
    Parameters
    ----------
    arr_lst : array list, arrays represent category codes, NaN is 
      represented as -1
    """
    arr = arr_lst[0]
    N = arr.shape[0]
    M = len(arr_lst)
    result = np.copy(arr)
    
    for n in range(N):
        if result[n] == -1:
            for j in range(1, M):
                v = arr_lst[j][n]
                if v != -1:
                    result[n] = v
                    break
    return result


def first_non_null_datetime(df):
    """Specialized first-non-null implementation for dataframes with 
    all datetime64 columns.
    """
    values = df.values
    rows, cols = (~np.isnat(values)).nonzero()
    rows, ix = np.unique(rows, return_index=True)
    cols = cols[ix]
    nnz = values[rows, cols]
    return pd.Series(nnz, index=df.index[rows]).reindex(df.index)


def first_non_null_cat_column(df):
    """Specialized version of first_non_null_column() for 
    categorical data frames.
    """
    ensure_only_cat_columns(df)
    dtype = common_cat_dtype(df)
    
    cols = List(df.iloc[:, i].astype(dtype).cat.codes.to_numpy() 
                for i in range(df.shape[1]))
    cat_codes = _first_non_null_cat_column(cols)
    values = pd.Categorical.from_codes(cat_codes, dtype=dtype)
    
    return pd.Series(values, index=df.index, name=df.columns[0])


def first_non_null_column(df):
    """Picks the first non-null value column-wise for 
    every row of data frame df. Returns a single-column
    data frame with identical index as df with the provided
    name. If name is None, the first column of df is used
    instead.
    """
    if df.ndim == 1:
        # nothitng to do
        return df
    
    if all_categorical(df):
        return first_non_null_cat_column(df)
    
    if not all_numeric(df):
        raise NotImplementedError(
            'Only numeric or categorical data is supported: '
            f'{df.columns.to_list()}.')
    
    # cast to double for new types such as pd.Int64Dtype
    array = df.astype('float64', copy=False).to_numpy()
    col = _first_non_null_column(array)
    return pd.Series(col, index=df.index, name=df.columns[0])


@numba.njit
def _unique_boolean_agg(index, values):
    if index.shape[0] != values.shape[0]:
        raise ValueError('Shapes do not match.')
        
    if index.shape[0] == 0:
        raise ValueError('Empty lists.')
        
    out_index = [index[0]]
    out_values = []
    val = values[0]
    prev_i = 0
    
    for i in range(1, index.shape[0]):
        if index[i] != index[prev_i]:
            out_index.append(index[i])
            out_values.append(val)
            val = values[i]
            prev_i = i
        else:
            val = val | values[i]
            
    out_values.append(val)
    return out_index, out_values


def unique_boolean_agg(series_or_df):
    """Optimized version of the following pandas transformation
    which works on boolean series/data frames:
    > series_or_df.groupby(level=0).max()
    
    Returns
    -------
    Boolean pandas Series/DataFrame with unique index.
    """
    if not isinstance(series_or_df.index, pd.Index):
        raise ValueError('Only 1D index is supported.')
        
    if not is_monotonic(series_or_df.index):
        series_or_df = series_or_df.sort_index()
        
    ix = series_or_df.index.to_numpy()
    values = series_or_df.to_numpy()
    ix, values = _unique_boolean_agg(ix, values)
    
    if series_or_df.ndim == 1:
        return pd.Series(values, index=ix, name=series_or_df.name)
    else:
        return pd.DataFrame(values, index=ix, columns=series_or_df.columns)
    
    
def find_any_values(df, val_list, aggregate=True):
    """Pattern matching / value equality for data of various data types. 
    Categorical data is dispatched to find_values() while numerical data is 
    processed for using numeric value equality (all values in val_list must
    be numbers). 
    """
    if df.dtypes.map(pd.api.types.is_numeric_dtype).all():
        # all patterns must convert to numbers
        num_values = np.array([float(v) for v in val_list])
        df = df.apply(lambda s: s.isin(num_values))
        return df.max(axis=1) if aggregate else df
    else:
        return find_values(df, val_list, aggregate)


def find_values(df, val_list, aggregate=True):
    """Returns a series indexed like df, which has a value True 
    when one of the columns in df contains a value from val_list
    in the corresponding row, or a boolean data frame when aggregate 
    is False.

    Partial matching support:
        Each value in val_list may end with '*', meaning that the
        matching will be done on the string preceding the asterisk
        (regardless of its contents).
        
    Parameters
    ----------
    df : the data frame with all categorical columns
    val_list : list of strings to search for (each possibly ending with *)
    aggregate : when False, return a boolean data frame of the same shape
      as df, indicating where any of the listed patterns was found.
    """
    ensure_only_cat_columns(df)

    simple_vals = [v for v in val_list if not v.endswith('*')]
    prefixes = [re.escape(v[:-1]) for v in val_list if v.endswith('*')]
    parts = []
    
    def maybe_aggregate(df):
        return df.max(axis=1) if aggregate else df

    if simple_vals:
        part = df.apply(lambda s: s.isin(val_list))
        parts.append(maybe_aggregate(part))

    if prefixes:
        # fuse all prefixes (if many) into a single regex for speed
        pattern = f'^({"|".join(prefixes)})'
        part = df.apply(lambda s: s.str.match(pattern).fillna(False))
        parts.append(maybe_aggregate(part))

    if len(parts) == 2:
        return parts[0] | parts[1]
    else:
        return parts[0]
    
    
@numba.njit
def _masked_select(values, mask):
    N, M = mask.shape
    result = []
    ix = []
    for i in range(N):
        found = False
        for j in range(M):
            if mask[i, j]:
                if not found:
                    row = [values[0, 0] for _ in range(0)]  # hint type inference
                    found = True
                row.append(values[i, j])
        if found:
            row.sort()
            result.append(np.array(row, values.dtype))
            ix.append(i)
    return result, ix


@numba.njit
def _series_to_rows(values, ix):
    """Converts a tall representation of a matrix to the list of rows
    format.
    """
    result = []
    index = []
    current = 0
    # numba siliness - empty list definition for the type inferencer
    row = [v for v in values[:0]]  
    for i in range(len(ix)):
        if ix[i] != ix[current]:
            # a new value
            index.append(ix[current])
            row.sort()
            result.append(np.array(row, values.dtype))
            current = i
            row = [v for v in values[:0]]  # new empty list
        row.append(values[i])
    # add the current data
    row.sort()
    result.append(np.array(row, values.dtype))
    index.append(ix[current])
    return result, index


@numba.njit
def _time_before(a_rows, b_rows, result, find_min,
                 _empty = np.timedelta64('NaT', 'ns')):
    for i in range(len(result)):
        result[i] = find_adjacent_diffs_before(
            a_rows[i], b_rows[i], _empty, find_min)
    return result


@numba.njit
def _time_between(a_rows, b_rows, result, find_min,
                 _empty = np.timedelta64('NaT', 'ns')):
    for i in range(len(result)):
         ab = find_adjacent_diffs_before(
            a_rows[i], b_rows[i], _empty, find_min)
         ba = find_adjacent_diffs_before(
            b_rows[i], a_rows[i], _empty, find_min)
         if find_min:
             res = ab if ab < ba else ba
         else:
             res = ab if ab > ba else ab
         result[i] = res
    return result


@numba.njit
def _rows_chose_one(rows, n, result):
    for i in range(len(result)):
        result[i] = rows[i][n]
    return result


class RowList:
    """A data structure that is a sparse representation of the operation of
    selecting a set of values from a data frame with a boolean mask. RowList
    stores the indexes of rows that have one or more matching value and, for
    each such an index value, the matching values are stored in a *sorted* numpy
    array. Also, RowLists store their rows with *sorted* indexes.
    
    RowList is, thus, akin to a sparse matrix (CSR) but it does not store column
    indices, treating the values simply as sequences. This is useful with the
    storage format of value lists in UKB multi-instance data fields.
    
    The data structure implements merging of multiple RowLists (which would come
    from parallel evaluation on multiple data set partitions).
    """
    def __init__(self, data=None, mask=None, *, rows=None, ix=None, dtype=None):
        """Creates a new RowList from either the data (df, mask) or 
        precalculated internal representations (rows, ix, dtype). The data 
        arguments can be either dataframes indicating WIDE data (so a unique
        index and multiple values in a row) or they can be series, indicating 
        TALL data (so non-unique index, one value per row).
        
        Parameters
        ----------
        data : data frame or series,
          Values to extract. If data frame, all columns must be of the same 
          type. When Series, it is considered tall format.
        mask : a boolean data frame or series,
          The mask to apply to the data (must be of the same dimension and with
          an integer index
        """
        if data is not None:
            if isinstance(data, pd.DataFrame):
                self._init_from_wide_mask(data, mask)
            else:
                self._init_from_tall_mask(data, mask)
        else:
            self._init_from_rows(rows, ix, dtype)
        # sorted index facilitates other operations
        self.data.sort_index(inplace=True)
            
    def _init_from_tall_mask(self, data, mask):
        self.dtype = data.dtype
        values = data[mask].sort_index()
        rows, ix = _series_to_rows(values.to_numpy(), values.index.to_numpy())
        self.data = pd.Series(rows, index=ix)
        
    def _init_from_wide_mask(self, df, mask):
        if len(df.dtypes.drop_duplicates()) > 1:
            raise ValueError('RowList data columns must have the same type')
        value_arr = df.to_numpy()
        mask_arr = mask.to_numpy()
        if value_arr.shape != mask_arr.shape:
            raise ValueError('df and mask must have the same shape.')
        if mask_arr.dtype != 'bool':
            raise ValueError('Mask is not boolean')
        rows, ix = _masked_select(value_arr, mask.values)
        self.data = pd.Series(rows, index=df.index[ix], dtype='O')
        self.dtype = value_arr.dtype
        
    def _init_from_rows(self, rows, ix, dtype):
        self.data = pd.Series(rows, index=ix)
        self.dtype = dtype or rows[0].dtype
        
    @property
    def index(self):
        return self.data.index
        
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.index)
    
    def __getitem__(self, item):
        return self.data[item]
    
    def items(self):
        return self.data.items()
    
    def _time_diffs(self, b, time_before=True, find_min=True):
        if self.dtype != 'datetime64[ns]' or b.dtype != 'datetime64[ns]': 
            raise ValueError('Time diff operators require datetime row lists.')
        
        dtype = np.timedelta64('NaT', 'ns') 
        common_ix = self.index.intersection(b.index)
        if common_ix.empty:
            return pd.Series([], index=common_ix, dtype=dtype)
        compute_diffs = _time_before if time_before else _time_between
        result = compute_diffs(
            List(self.data[common_ix]),
            List(b[common_ix]),
            np.empty(len(common_ix), dtype),
            find_min)
        return pd.Series(result, index=common_ix)
    
    def time_before(self, other, find_min=True):
        """Minimum time among events when an event in self directly precedes
        an event in other. Both self and other must be datetimes.
        For details check docs of find_adjacent_diffs_before().
        """
        return self._time_diffs(other, True, find_min)
    
    def time_between(self, other, find_min=True):
        """Minimum time among events when an event in self directly precedes
        an event in other *or vice-versa*. Both self and other must be 
        datetimes. For details check docs of find_adjacent_diffs_before().
        """
        return self._time_diffs(other, False, find_min)
    
    def min(self):
        """Returns a pd.Series with earliest dates for each row.
        """
        result = np.empty(len(self), 'datetime64[ns]')
        # note, the lists are sorted
        if len(result) > 0:
            _rows_chose_one(List(self.data), 0, result)
        return pd.Series(result, index=self.index)
    
    def max(self):
        """Returns a pd.Series with latest dates for each row.
        """
        result = np.empty(len(self), 'datetime64[ns]')
        # note, the lists are sorted
        if len(result) > 0:
            _rows_chose_one(List(self.data), -1, result)
        return pd.Series(result, index=self.index)
    
    @staticmethod
    def _merge_row_lists(blocks, ix_list):
        to_merge = defaultdict(list)
        # first collect all rows for each key (there may be overlap)
        for i in range(len(blocks)):
            ix = ix_list[i]
            rows = blocks[i]
            for j in range(len(ix)):
                to_merge[ix[j]].append(rows[j])
        # now, merge multiple rows for those keys with more than one row
        new_ix = []
        new_rows = []
        for key, parts in to_merge.items():
            row = np.sort(np.concatenate(parts))
            if len(row):
                new_ix.append(key)
                new_rows.append(row)
        return new_rows, new_ix
    
    @classmethod
    def merge(cls, row_lists):
        """Merges multiple RowLists into a new one. The result will have a 
        unique index and if keys overlap, their corresponding rows will be
        merged (and sorted).
        
        Parameters
        ----------
        row_lists : the list of input row lists. They must all be of the 
          same dtype.
          
        Returns
        ------- 
        A new RowList that is the sum of all the inputs.
        """
        ix_lst = []
        blocks = []
        dtype = None
        for rl in row_lists:
            ix_lst.append(rl.data.index.to_numpy())
            blocks.append(rl.data.to_numpy())
            if dtype is None:
                dtype = rl.dtype
            else:
                if dtype != rl.dtype:
                    raise ValueError('Row lists have different dtypes.')
        rows, ix = cls._merge_row_lists(blocks, ix_lst)
        return RowList(rows=rows, ix=ix, dtype=dtype)
    

@numba.njit
def find_adjacent_diffs_before(a, b, _empty, find_min=True):
    """For two non-empty and sorted arrays a and b, finds min/max
    of a[i] - b[j] for all i, j such that a[i] and b[j] are adjacent
    in a sorted list of all elements from both a and b (and a[i] comes
    before b[j]).
    
    Parameters
    ----------
    a, b : two 1D arrays to compare, of the same type, may differ in lenght.
    """
    result = _empty
    success = False
    i = 0
    j = 0
    while True:
        if a[i] <= b[j]:
            # a precedes b, so store the value by 
            # calculating minimum in-place
            new_val = b[j] - a[i]
            if not success:
                result = new_val
                success = True
            elif (find_min and new_val < result
                  or find_min == False and new_val > result):
                result = new_val
              
            # move to the next a, finish otherwise
            if i < len(a) - 1:
                i += 1
            else:
                break
        elif j < len(b) - 1:
            j += 1
        else:
            # nothing left to compare with
            break
    return result


def common_cat_dtype(categoricals):
    """Creates a pd.CategoricalDType that combines categories from all 
    categoricals. 
    
    Parameters
    ----------
    categoricals : either pd.DataFrame or list of pd.Series
    
    Returns
    -------
    pd.CategoricalDType with all categories contained in the inputs
    """
    ensure_only_cat_columns(categoricals)

    if isinstance(categoricals, pd.DataFrame):
        categoricals = [col for _, col in categoricals.items()]
        
    categories = np.unique(np.concatenate([
        c.cat.categories.to_numpy() for c in categoricals]))

    # the new type must be ordered to ensure correct category dtype 
    # conversions elsewhere
    return pd.CategoricalDtype(categories, ordered=True)


def ensure_only_cat_columns(df):
    non_cat_cols = df.dtypes != 'category'
    if non_cat_cols.any():
        bad_cols = str(df.columns[non_cat_cols].to_list())
        if len(bad_cols) > 200:
            bad_cols = bad_cols[:200] + '...'
        raise ValueError(f'All columns must be categorical: {bad_cols}')


def as_common_cat(df):
    """Converts all columns in df from independent categorical
    dtypes to a single categorical dtype that contains all
    categories found in df
    
    Parameters
    ----------
    df : pd.DataFrame with all columns of categorical dtype
    """
    dtype = common_cat_dtype(df)
    return pd.DataFrame({name: col.astype(dtype) for name, col in df.items()})


def find_group_sizes(counts, max_scaling=None, ref_cost=2):
    """A utility function that reads a 2-column data frame with sample
    counts for each group (the index) for each of two data sets.
    The method uses optimization to find best scaling factor for downsampling
    one of the datasets and applies corrections for all groups for which there
    are not enough samples to satisfy the required proportions.
    """
    # since column 0 is to be resampled, make sure column 1 does not 
    # have more samples in any group
    count_values = np.vstack([counts[0], np.minimum(counts[0], counts[1])])
    
    if max_scaling is None:
        # a rough approximation of max scaling factor
        totals = count_values.sum(axis=1)
        if totals[1] == 0:
            raise ValueError(
                'Cannot calculate group sizes due to case count of 0.')
        max_scaling = totals[0] / totals[1]
        
    if max_scaling <= 0:
        raise ValueError('Scaling must be > 0.')
            
    def opt_counts(scale):
        """Cost function for a scaling factor.
        ref_cost - the cost of dropping a sample from the reference when it
          can be traded off for more samples in the main data set.
        """
        diff = count_values[1] * scale - count_values[0]
        cost = np.where(diff > 0, diff * ref_cost, diff)
        return np.abs(cost).sum()
    
    opt = minimize_scalar(opt_counts,
                          method='bounded',
                          bounds=(0, max_scaling))
    if not opt.success:
        raise ValueError(f'Cannot find scaling factor: {count_values}')

    # general scaling brings both data sets to similar sizes
    # once comparable, trim individual group sizes for exact matching
    general_scale = opt.x
    
    # desired group sizes for the sampled data
    sizes_0 = np.minimum(np.round(count_values[1] * general_scale), 
                         count_values[0])
    # desired group sizes for the reference data (if some need downsampling)
    sizes_1 = np.minimum(np.round(sizes_0 / general_scale), 
                         count_values[1])
    
    sizes = np.vstack([sizes_0, sizes_1]).astype('int')
    return pd.DataFrame(sizes.T, index=counts.index)


def stratified_sample(df, reference, factors, *, 
                      size_factor=None, 
                      df_scores=None,
                      ref_downsampling_cost=2):
    """Resamples df and reference data frames in such a way as to have the same 
    sample proportions with respect to a number of factors (columns) in both 
    data frames. A scaling factor can also be provided to limit the overall size
    of resampled data frame to a multiple (or fraction) of the reference.
    
    Parameters
    ----------
    df : the data frame to resample.
    reference : a data frame that defines the desired distribution of factors.
    factors : list of strings
      Column names or a column name in both df and reference data frames. 
    size_factor : float
      Size ratio of the resulting sub-sampling of df to the reference. Useful, 
      when the resampled df size needs to be related to the size of the 
      reference (like no more than 2x the size). The default is None, meaning 
      that only group proportions are maintained.
    df_scores : pd.Series, numeric, optional
      When provided, df records are removed in the order of decreasing scores
      instead of random down-sampling.
    ref_downsampling_cost : int, optional 
      Relative cost of dropping a case vs dropping a control to satisfy required
      size factor. When > 1, cases will tend to be preserved by sampling. Use 
      larger values to ensure that cases are never dropped (>= 10).
    Returns
    -------
    Concatenation of resampled df and reference data frames.
    """
    if df_scores is not None:
        common_ix = df.index.union(reference.index)
        df_scores = df_scores.reindex(common_ix)\
            .fillna(df_scores.min())\
            .sort_index()
        if df_scores.isna().all():
            raise ValueError('All scores cannot be NA')
        
    def select_samples(index, n):
        """Selects n samples to retain from index.
        """
        if n == len(index):
            return index
        if df_scores is None:
            return RND().choice(index, n, replace=False)
        else:
            # n samples with lowest scores
            return df_scores.loc[index].sort_values().index[:n]
        
    df_groupby = df.groupby(factors, observed=True)
    ref_groupby = reference.groupby(factors, observed=True)
    
    counts = pd.concat([df_groupby.size(), ref_groupby.size()], 
                       keys=[0, 1], axis=1).fillna(0)
    
    # check if there's any data to be returned (expecting at least one
    # row to have non zero values everywhere).
    if (counts.product(axis=1) == 0).all():
        return pd.DataFrame(columns=df.columns)
    
    parts = []
    grp_sizes = find_group_sizes(counts, size_factor, ref_downsampling_cost)
    for grp_id, part_df, groupby in zip([0, 1], 
                                    [df, reference], 
                                    [df_groupby, ref_groupby]):
        ix_lst = []
        for group, ix in groupby.groups.items():
            n = grp_sizes.loc[group, grp_id]
            if n > 0:
                ix_lst.append(select_samples(ix, n))
        parts.append(part_df.loc[np.sort(np.hstack(ix_lst))])
        
    return pd.concat(parts)
    

def check_proportions_equal(counts):
    """Runs Fisher's exact test on the contingency table. In case of
    large counts, Chi2 test is used to make the processing faster.
    
    Parameters
    ----------
    counts : contingency table, numpy array
    
    Returns
    -------
    p value of the test
    """
    if counts.mean() <= 500:
        _, pval = fisher_exact(counts, alternative='two-sided')        
    else:
        _, pval, _, _ = chi2_contingency(counts)
    return pval


def ensure_gender_proportions(cohort, gender, alpha=.05, scores=None, 
                              case_drop_cost=2):
    """Resamples the cohort if Fisher test indicates different gender 
    distributions in cases and controls. Resampling is performed to make
    the controls match the cases in gender proportions.
    
    Paramterers
    -----------
    cohort : pd.Series, integer
      1s indicate cases and 0s the controls
    gender : pd.Series
      Gender information for each cohort member, cannot have NAs
    scores : pd.Series, numeric
      Optional sample scores that define the order of member removal (instead of
      random sub-sampling). Members with higher scores are removed first.
    case_drop_cost : int, optional
      The cost of downsampling from the set of cases: Values > 1 will penalize
      downsampling cases, use a larger value to prevent that from happening at
      all. 
    """
    if any(not isinstance(arg, pd.Series) for arg in (cohort, gender)):
        raise ValueError('Expecting pd.Series.')
    
    cohort = cohort.dropna()
    df = pd.concat([cohort, gender.reindex(cohort.index)], 
                   keys=['cohort', 'gender'], 
                   axis=1, copy=False)
        
    if sorted(df['cohort'].unique()) != [0, 1]:
        raise ValueError('Expecting binary integer series as cohort.')
        
    if df['gender'].isna().any():
        raise ValueError('Gender information not available for some subjects.')
    # shape: len(distinct genders) * len([controls, cases])
    counts = df.groupby(['cohort', 'gender'], observed=True)\
        .size()\
        .unstack('cohort', fill_value=0)
    if len(counts) == 1:
        # only one gender in both cases and controls - nothing to resample 
        result = cohort
    elif len(counts) == 2:
        # check for statistical equality of gender proportions
        pval = check_proportions_equal(counts.values)
        if pval > alpha:
            # nothing to do - the proportions are likely to be similar
            result = cohort
        else:
            # resample whichever is larger - either cases or controls
            to_subsample = df['cohort'] == counts.columns[counts.sum().argmax()]
            result = stratified_sample(
                df[to_subsample], 
                df[~to_subsample], 
                ['gender'],
                df_scores=scores,
                ref_downsampling_cost=case_drop_cost)
    else:
        raise ValueError('Only 1 or 2 distinct gender values are permitted.')
    return cohort.reindex(result.index)
    
    
def as_dataframes(objects):
    """Utility method that accepts a list of pandas series and/or data frame
    objects and casts all series to data frames. Objects of other types are 
    not accepted.
    """
    result = []
    for obj in objects:
        if isinstance(obj, pd.Series):
            result.append(obj.to_frame())
        elif isinstance(obj, DataFrame):
            result.append(obj)
        else:
            raise ValueError(f'Expected pd.Series or pd.DataFrame, got {obj}.')
    return result


def with_comas(lst):
    """Returns a string with coma-separated values of the sorted lst.
    """
    if len(lst) == 0:
        return '[none]'
    elif len(lst) == 1:
        return list(lst)[0]
    else:
        return ', '.join(sorted(map(str, lst)))
