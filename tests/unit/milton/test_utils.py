import pytest
import numpy as np 
import pandas as pd
from pandas.testing import assert_series_equal
from numbers import Number
from milton.utils import *


def test_v_concat_with_categories_for_df():
    df_parts = []
    for i in range(3):
        m = 10 ** i  # multiplier
        ix = pd.Index(m * np.arange(5))

        df_parts.append(pd.DataFrame({
            # first few non-categorical columns
            'c0': m * np.arange(5),  # integers
            'c1': m * np.full(5, 1.0, dtype='float'),
            'c2': pd.Series(m * np.arange(5), dtype='string', index=ix),

            # categorical columns
            'c3': ('cat_' + pd.Series(m * np.arange(5),
                                      dtype='string',
                                      index=ix)).astype('category'),

            'c4': ('cat_' + pd.Series(m * np.arange(5, dtype='float'),
                                      dtype='string',
                                      index=ix)).astype('category')
        }))

    # the 'keys' keyword arg tests whether the extra pd.concat() args
    # are handled correctly
    result = v_concat_with_categories(df_parts, keys=['A', 'B', 'C'])
    basic_concat = pd.concat(df_parts, keys=['A', 'B', 'C'])

    # non-categorical columns will behave in a standard way
    pd.testing.assert_frame_equal(
        basic_concat[['c0', 'c1', 'c2']],
        result[['c0', 'c1', 'c2']])

    assert (result[['c3', 'c4']].dtypes == 'category').all()

    pd.testing.assert_frame_equal(
        basic_concat[['c3', 'c4']].astype('str'),
        result[['c3', 'c4']].astype('str'))


def test_v_concat_with_categories_for_series():
    keys = list('ABCD')
    parts = []
    for i in range(3):
        m = 10 ** i  # multiplier
        ix = pd.Index(m * np.arange(5))
        part = ('cat_' + pd.Series(m * np.arange(5),
                                   dtype='string',
                                   index=ix)).astype('category')
        parts.append(part)
    # add a series full of NAs as a special case
    ix = pd.Index(10**4 * np.arange(5))
    parts.append(pd.Series(np.nan, ix, dtype='string').astype('category'))

    result = v_concat_with_categories(parts, keys=keys)
    basic_concat = pd.concat(parts, keys=keys)

    assert isinstance(result, pd.Series)
    assert result.dtype == 'category'
    assert_series_equal(basic_concat.astype('string'), result.astype('string'))

    # the same as above but with parts being non-categorical
    non_cat_parts = [s.astype('string') for s in parts]
    result = v_concat_with_categories(non_cat_parts, keys=keys)
    basic_concat = pd.concat(non_cat_parts, keys=keys)

    assert_series_equal(basic_concat.astype('string'), result.astype('string'))

    
def test_first_non_null_column_floats():
    inp = pd.DataFrame([
        [np.nan, np.nan, 1],
        [2, np.nan, np.nan],
        [np.nan, 3, np.nan],
        [np.nan, np.nan, np.nan]
    ], columns=['A']*3)

    for df in [inp, inp.astype('string').astype('category')]:
        outp = first_non_null_column(df).astype('float')
        assert_series_equal(pd.Series([1, 2, 3, np.nan], name='A'),
                            outp)
        
    
def test_first_non_null_column_categorical():
    inp = pd.DataFrame([
        [pd.NA, 'a', 'b'],
        ['a', pd.NA, 'c'],
        [pd.NA] * 3,
        [pd.NA, pd.NA, 'a']
    ], dtype='category', columns=['A']*3)
    outp = first_non_null_column(inp)
    expected = pd.Series(pd.Categorical(['a', 'a', pd.NA, 'a'], 
                                        categories=['a', 'b', 'c'],
                                        ordered=True), 
                         name='A')
    assert_series_equal(expected, outp)

    
def test_unique_boolean_agg_series():
    series = pd.Series(
        data=[False, True, False, True, False],
        index=[1, 3, 2, 2, 1])
    
    result = unique_boolean_agg(series)
    expected = pd.Series([False, True, True], index=[1, 2, 3])
    assert result.equals(expected)
    
    
def test_unique_boolean_agg_dataframe():
    df = pd.DataFrame([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype='bool', index=[0, 1, 0])
    
    result = unique_boolean_agg(df)
    expected = pd.DataFrame([
        [1, 1, 1],
        [0, 1, 0]
    ], dtype='bool', index=[0, 1])
    assert result.equals(expected)
    
    
def test_find_values_simple():
    df = pd.DataFrame({
        'A': ['a', 'b', 'c'],
        'B': ['x', 'y', 'z'],
        'C': ['p', 'q', 'r']
    }).astype('category')
    out = find_values(df, ['a', 'y'])
    assert out.dtype == 'bool'
    assert isinstance(out, pd.Series)
    assert out.to_list() == [True, True, False]


def test_find_values():
    df = pd.DataFrame([
        ('xxx',     pd.NA,      'Code3'),
        (pd.NA,     'a',        'Code4'),
        (pd.NA,     'Code2',    'b'),
        ('Code1',   pd.NA,      'Code5'),
        ('b',       'b',        pd.NA),
        (pd.NA,     'Code2-1',  'xxx'),
        ('Code2-2',  pd.NA,     'yyy')
    ], dtype='category')

    result = find_values(df, ['Code*'])
    assert isinstance(result, pd.Series)
    assert result.dtype == 'bool'

    # simple matching of values (no prefix matching)
    result = find_values(df, ['a', 'b']).to_list()
    assert result == [False, True, True, False, True, False, False]

    # prefix matching mixed with simple matching
    result = find_values(df, ['Code1', 'Code2*']).to_list()
    assert result == [False, False, True, True, False, True, True]

    # prefix-only matching
    result = find_values(df, ['Code1*', 'Code2*']).to_list()
    assert result == [False, False, True, True, False, True, True]

    result = find_values(df, ['Code*']).to_list()
    assert result == [True, True, True, True, False, True, True]


def test_find_values_avoids_regex_capture():
    """find_values uses internaly regex matching for patterns with asterisk. 
    Make sure, this is handled correctly.
    """
    df = pd.DataFrame([
        ('abc',     'def'),
        ('a.cXX',     pd.NA),
        (pd.NA,     'd.fXX'),
    ], dtype='category')

    # the dot char should be matched explicitly (not as regex any-char)
    result = find_values(df, ['a.c*', 'd.f*']).to_list()
    assert result == [False, True, True]


def test_find_values_requires_categoricals():
    df = df = pd.DataFrame([
        ('xxx', pd.NA, 'Code3'),
        (pd.NA, 'a', 'Code4')])
    with pytest.raises(ValueError):
        find_values(df, ['a', 'b'])


def test_remap_categories():
    categories = pd.Series(
        pd.Categorical(['a', 'a', 'b', 'c', 'c']),
        name='a series',
        index=10 * (1 + np.arange(5)))

    # non-unique mapping
    out = remap_categories(categories, {'a': 'A', 'b': 'B', 'c': 'A'})
    expected = pd.Series(
        pd.Categorical(['A', 'A', 'B', 'A', 'A']),
        name=categories.name,
        index=categories.index)
    assert_series_equal(out, expected)

    # mapping with missing categories
    out = remap_categories(categories, {'a': 'A', 'b': 'B'})
    expected = pd.Series(
        pd.Categorical(['A', 'A', 'B', pd.NA, pd.NA]),
        name=categories.name,
        index=categories.index)
    assert_series_equal(out, expected)


def test_as_common_cat():
    df = pd.DataFrame({
        'A': pd.Categorical(['a', 'b', 'c']),
        'B': pd.Categorical(['d', 'e', 'f'])})

    out = as_common_cat(df)
    assert (df.dtypes == 'category').all()
    assert out['A'].dtype == out['B'].dtype
    assert(out['A'].dtype.categories == list('abcdef')).all()
    
    
@pytest.mark.parametrize('is_tall', [False, True])
def test_row_lists(is_tall):
    """RowList functionality for wide tables (multiple values in a row, unique
    index).
    """
    data = pd.DataFrame([
        [1, 4, 5],
        [2, 2, 1],
        [3, 2, 1]
    ])
    mask = pd.DataFrame([
        [False, False, False],  # filtered out
        [True, True, False],
        [True, False, True]
    ])
    if is_tall:
        # reshape the data but expect the same results
        data = data.stack().reset_index(level=1, drop=True)
        mask = mask.stack().reset_index(level=1, drop=True)
        
    rl = RowList(data, mask)
    assert rl.dtype == np.int64
    # check rows which had some matching values - 0 is missing
    assert list(rl) == [1, 2]
    assert np.equal(rl[1], np.array([2, 2], dtype='int64')).all()
    assert np.equal(rl[2], np.array([1, 3], dtype='int64')).all()
    
    
def test_row_lists_tall():
    """A dedicated test for the tall data format.
    """
    data = pd.Series(range(10, 0, -1), index=[4, 4, 1, 2, 6, 4, 2, 1, 5, 6])
    # select the first 5 values, so 4 unique IDs
    mask = pd.Series([True] * 5 + [False] * 5, index=data.index)
    rl = RowList(data, mask)
    assert len(rl) == 4
    assert list(rl) == [1, 2, 4, 6]
    assert list(rl[1]) == [8]
    assert list(rl[2]) == [7]
    assert list(rl[4]) == [9, 10]  # sorted
    assert list(rl[6]) == [6]
    
    
def DA(*lst):
    """Shorthand for an array of dates.
    """
    return np.array(lst, dtype='datetime64[ns]')


def DL(*lst):
    """Shorthand for a list of dates.
    """
    return list(DA(*lst))
    
    
def test_row_list_merge():
    NA = pd.NA
    ix_wide = [1, 2, 3]
    rl_wide = RowList(
        pd.DataFrame([
            ['2020-05-05', NA, '2020-01-01', NA],
            ['2020-02-02', NA, NA, NA],
            ['2020-07-07', '2020-06-06', '2020-06-06', '2020-03-03'],
        ], index=ix_wide, dtype='datetime64[ns]'),
        pd.DataFrame([
            [False, False, True, False],
            [False] * 4,  # no. 2 is filtered out
            [False, True, True, True]
        ], index=ix_wide, dtype='bool'),
    )
    ix_tall = [2, 2, 5, 1, 5]
    rl_tall = RowList(
        pd.Series(
            ['2021-01-01', '2021-02-02', '2010-10-10', '2020-04-04', '2011-11-11'],
            index=ix_tall,
            dtype='datetime64[ns]'),
        pd.Series(
            # no. 1 is filtered out
            [True, False, False, False, True],
            index=ix_tall,
            dtype='bool'))
    
    result = RowList.merge([rl_wide, rl_tall])
    # despite the filtering out, all IDs appear in the union
    assert list(result) == [1, 2, 3, 5]
    assert list(result[1]) == DL('2020-01-01')
    assert list(result[2]) == DL('2021-01-01')
    assert list(result[3]) == DL('2020-03-03', '2020-06-06', '2020-06-06') # sorted
    assert list(result[5]) == DL('2011-11-11')
   
    
def test_row_list_time_before():
    # note, that the index values define which rows will be compared against
    # note also, that this form of row list definition requires the arrays
    # to be already sorted
    a = RowList(
        rows=[
            DA('2020-03-01', '2020-07-07', '2020-10-05'),
            DA('2020-03-01', '2020-08-11'),
            DA('2020-12-30'),  # no match
        ],
        ix=[5, 100, 77])
    b = RowList(
        rows = [
            DA('2020-03-15'),
            DA('2020-01-22', '2020-05-05', '2020-11-11', '2020-12-12'), # no match
            DA('2020-01-01', '2020-01-10', '2020-10-10')
        ],
        ix=[100, 4, 5])
    
    outp = a.time_before(b)
    expected = pd.to_timedelta(
        pd.Series([5, 14],  # durations in days
                  index=[5, 100]),   # index must be sorted
        unit='D')
    pd.testing.assert_series_equal(outp, expected)
    
    
def test_row_list_min_max():
    rl = RowList(
        rows=[
            DA('2020-03-01', '2020-07-07', '2020-10-05'),
            DA('2020-03-01', '2020-08-11'),
            DA('2020-12-30'),  # no match
        ],
        ix=[5, 100, 77])
    expected = pd.Series([
        '2020-03-01',
        '2020-12-30',
        '2020-03-01',
    ], index=[5, 77, 100], dtype='datetime64[ns]')
    pd.testing.assert_series_equal(rl.min(), expected)
    
    expected = pd.Series([
        '2020-10-05',
        '2020-12-30',
        '2020-08-11',
    ], index=[5, 77, 100], dtype='datetime64[ns]')
    pd.testing.assert_series_equal(rl.max(), expected)
    
def test_find_adjacent_diffs_before():
    # A interleaved with B
    a = np.array([1, 3, 8])
    b = np.array([5, 6, 12])
    assert 2 == find_adjacent_diffs_before(a, b, _empty=-1)
    
    # short arrays
    a = np.array([20])
    b = np.array([30])
    assert 10 == find_adjacent_diffs_before(a, b, _empty=-1)
    
    # A never preceeds B
    a = np.array([30, 35, 40])
    b = np.array([20])
    assert -1 == find_adjacent_diffs_before(a, b, _empty=-1)
    
    # all of A preceeds all of B
    a = np.array([1, 3, 5])
    b = np.array([10, 20, 30])
    assert 5 == find_adjacent_diffs_before(a, b, _empty=-1)
    

def make_test_cohorts(grp_sizes, ctl_scales=1):
    """Constructs two cohort data frames (cases/controls) along with 
    grouping factor columns. The data frames will be of shape: 
    (value, fact0, fact1,...) where value column is 1 for cases and 
    0 for controls. The factN columns correspond to the index values in 
    the grp_sizes series, and N >= 1 when the keys are tuples (of the 
    same length).
    
    Parameters
    ----------
    grp_sizes : pd.Series from grouping factor values to the corresponding
      numbers of samples in the result. 
    ctl_scales : pd.Series of the same structure as grp_sizes - the growth/shrink 
      factors for the case groups - applies to the controls. Can also be 
      a single integer for uniform treatment.
    """
    if isinstance(ctl_scales, Number):
        ctl_scales = pd.Series(ctl_scales, index=grp_sizes.index)
    else:
        assert len(ctl_scales) == len(grp_sizes)
    
    cases = []
    controls = []
    
    for grp, size in grp_sizes.items():
        for cohort, value, fact in [(cases, 1, 1), 
                                    (controls, 0, ctl_scales[grp])]:
            ix_values = [grp] * int(size * fact)
            if isinstance(grp, tuple):
                names = [f'fact{i}' for i in range(len(grp))]
                ix = pd.MultiIndex.from_tuples(ix_values, names=names)
            else:
                ix = pd.Index(ix_values, name='fact0')
            cohort.append(pd.Series(value, index=ix))
            
    case_df = pd.concat(cases).to_frame('value').reset_index()
    ctl_df = pd.concat(controls).to_frame('value').reset_index()
    ctl_df.index = pd.Index(np.arange(len(case_df), len(case_df) + len(ctl_df)))
    return case_df, ctl_df


def assert_group_sizes(df, expected_sizes):
    """Checks if df.groupby(['frac0, ...']).size() has the same values as 
    expected_sizes. See make_test_cohorts() for details.
    """
    if isinstance(expected_sizes, dict):
        expected_sizes = pd.Series(expected_sizes)
        
    # ensure the specs are consistent
    key_lengths = expected_sizes.index\
        .map(lambda v: len(v) if isinstance(v, tuple) else 1)
    assert len(key_lengths.unique()) == 1
    dim = key_lengths.unique()[0]
    
    grp_sizes = df.groupby([f'fact{i}' for i in range(dim)]).size()
    assert grp_sizes.to_dict() == expected_sizes.to_dict()
    assert not df.index.duplicated().any()
    return expected_sizes


def assert_equal_proportions(df):
    """Checks equality of proportions of samples across 
    sub-groups between cases and controls.
    """
    props = df.groupby(df.columns.to_list()).size()\
        .unstack(level='value')\
        .pipe(lambda s: s/s.sum())
    assert np.allclose(props[1], props[0], atol=1e-4)
    

def check_group_sizes(counts, *, max_scaling=None, equals=None, ref_downsampled=False):
    counts = pd.DataFrame(counts)
    
    if isinstance(max_scaling, list):
        scalings = max_scaling
    else:
        scalings = [max_scaling]
        
    for max_scaling in scalings:
        result = find_group_sizes(counts, max_scaling)
        if equals:
            assert result.equals(pd.DataFrame(equals))
        assert (result <= counts).all().all()
        assert (result >= 0).all().all()

        # check equality of proportions
        props = result / result.sum() 
        assert np.allclose(props[0], props[1], atol=1e-2)
        if not ref_downsampled:
            assert np.allclose(props[1], counts[1]/counts[1].sum(), atol=1e-2)

    
def test_stratified_sample_testing_utils():
    """self-check of the testing utility code.
    """
    sizes = pd.Series({'male': 50, 'female': 60})
    cases, ctls = make_test_cohorts(sizes, ctl_scales=2)
    assert_group_sizes(cases, sizes)
    assert_group_sizes(ctls, 2 * sizes)
    

def test_find_groups_sizes():
    # fixed scaling factor
    check_group_sizes(
        counts = [
            [100, 20],
            [110, 40],
            [50, 10]], 
        equals = [
            [40, 20],
            [80, 40],
            [20, 10]],
        max_scaling=2)

    # no scaling factor - produce as much as possible
    check_group_sizes(
        counts = [
            [100, 20],
            [110, 40],
            [50, 10]], 
        equals = [
            [55, 20],
            [110, 40],
            [27, 10]],
        max_scaling=[None, 3, 5])
    
    # fractional scaling (< 1)
    check_group_sizes(
        counts = [
            [100, 20],
            [110, 40],
            [50, 10]], 
        equals = [
            [10, 20],
            [20, 40],
            [5, 10]],
        max_scaling=.5)

    check_group_sizes(
        counts = [
            [100, 20],
            [110, 40],
            [10, 10]],   # not enough samples for 2x scaling
        equals = [
            [40, 20],
            [80, 40],
            [10, 5]],   # reference is down-sampled
        max_scaling=2,
        ref_downsampled=True)

    check_group_sizes(
        counts = [
            [100, 20],
            [110, 40],
            [0, 10]],   # no samples in sub-group
        equals = [
            [40, 20],
            [80, 40],
            [0, 0]],   # reference is down-sampled
        max_scaling=2,
        ref_downsampled=True)

    # reference data has no samples in some groups
    check_group_sizes(
        counts = [
            [100, 20],
            [110, 40],
            [10, 0]],   
        equals = [
            [40, 20],
            [80, 40],
            [0, 0]],   # no samples in this group
        max_scaling=2)
    
    # there are more samples in the reference data, in each group
    check_group_sizes(
        counts = [
            [20, 100],
            [30, 200],
            [50, 150]], 
        equals = [
            [20, 20],
            [30, 30],
            [50, 50]],
        max_scaling=[None, 2, 5, 10], 
        ref_downsampled=True)
    
    # some reference groups have very few samples
    # (arises with phenotypes that affect mostly one gender)
    check_group_sizes(
        counts = [
            [100, 50],
            [200, 60],
            [120, 5],
            [200, 7]], 
        equals = [
            [100, 50],
            [120, 60],
            [10, 5],
            [14, 7]],
        max_scaling=[None, 2, 5, 10])
    
    
def test_find_group_sizes_no_data():
    """Exception should be rised when either cases or controls are all 0.
    """
    with pytest.raises(ValueError):
        find_group_sizes(pd.DataFrame([
            [10, 0],
            [20, 0],
            [30, 0]
        ]), max_scaling=None)
        
    with pytest.raises(ValueError):
        find_group_sizes(pd.DataFrame([
            [0, 10],
            [0, 30],
            [0, 60]
        ]), max_scaling=None)
        
    with pytest.raises(ValueError):
        find_group_sizes(pd.DataFrame([
            [0, 0],
            [0, 0],
            [0, 0]
        ]), max_scaling=None)
    

def test_stratified_sample_simple_ratios():
    """even scaling accross groups and controls have enough 
    samples to satisfy the requested ratios.
    """
    sizes = pd.Series({'male': 50, 'female': 60})
    cases, ctls = make_test_cohorts(sizes, ctl_scales={'male': 5, 'female': 5.75})

    # size factor of None means produce the maximum possible
    for sf in [1, 2, 3, 4, 5, None]:
        smp = stratified_sample(ctls, cases, ['fact0'], size_factor=sf)
        # nothing changes for the cases
        assert_group_sizes(smp.query('value == 1'), sizes)
        # number of controls has changed
        assert_group_sizes(smp.query('value == 0'), (sf or 5) * sizes)  
        
        
def test_stratified_sample_incomplete_ratios():
    """controls have a different male/female distribution,
    there may not be enough controls to satisfy the required
    size factor.
    """
    sizes = pd.Series({'male': 50, 'female': 60})
    cases, ctls = make_test_cohorts(sizes, ctl_scales={'male': 2, 'female': 1.5})
    
    # size factor small enough to resample
    smp = stratified_sample(ctls, cases, ['fact0'], size_factor=1)  
    # nothing changes for the cases
    assert_group_sizes(smp.query('value == 1'), sizes)
    # correctly downsampled
    assert_group_sizes(smp.query('value == 0'), 1 * sizes)  

    # larger size factor - not enough controls to satisfy
    for sf in [2, None]:
        # not enough females to satisfy
        smp = stratified_sample(ctls, cases, ['fact0'], size_factor=sf)  
        # nothing changes for the cases
        assert_group_sizes(smp.query('value == 1'), sizes)
        # minimum achievable size factor is 1.5
        assert_group_sizes(smp.query('value == 0'), 1.5 * sizes)  
        
        
def test_stratified_sample_not_enough_data():
    """There are sub-groups of controls with < 1 size factor
    """
    sizes = pd.Series({'male': 50, 'female': 60})
    cases, ctls = make_test_cohorts(sizes, ctl_scales={'male': 2, 'female': .5})
    
    for sf in [2, None]:  # expect the same results for None and 2
        # size factor small enough to resample
        smp = stratified_sample(ctls, cases, ['fact0'], size_factor=2)  
        assert_equal_proportions(smp)

        # females have been downsampled in cases
        assert_group_sizes(smp.query('value == 1'), {'male': 50, 'female': 30})
        # controls result in the same values
        assert_group_sizes(smp.query('value == 0'), {'male': 50, 'female': 30})
        
        
def test_stratified_sample_no_data_to_return():
    """All cases when an empty data frame should be returned.
    """
    sizes = pd.Series({'male': 50, 'female': 60})
    cases, ctls = make_test_cohorts(sizes)
    
    for sf in [1, 2, None]:  # expect the same results for None and 2
        smp = stratified_sample(ctls[:0], cases, ['fact0'])  
        assert smp.empty
        assert smp.columns.equals(ctls.columns)
        
        smp = stratified_sample(ctls, cases[:0], ['fact0'])  
        assert smp.empty
        assert smp.columns.equals(ctls.columns)
        
        # both data frame have totally different factor values (gender strings)
        # so no resampling is possible - nothing overlaps
        cases_different_fact = cases\
            .assign(fact0=lambda df: df['fact0'].str.upper())
        smp = stratified_sample(ctls, cases_different_fact, ['fact0'])  
        assert smp.empty
        assert smp.columns.equals(ctls.columns)

        
def test_stratified_sample_missing_groups_in_controls():
    """A sub-group is missing in controls - should be omitted from the result.
    """
    sizes = pd.Series({'male': 50, 'female': 60, 'dog': 20})
    cases, ctls = make_test_cohorts(sizes, ctl_scales={'male': 2, 'female': 2, 'dog': 0})
    
    for sf in [2, None]:  # expect the same results for None and 2
        # size factor small enough to resample
        smp = stratified_sample(ctls, cases, ['fact0'], size_factor=2)  
        assert_equal_proportions(smp)

        # in cases nothing changes except the dog category is no longer there
        assert_group_sizes(smp.query('value == 1'), {'male': 50, 'female': 60})
        # there 2x more controls and no dogs
        assert_group_sizes(smp.query('value == 0'), {'male': 2 * 50, 'female': 2 * 60})  
        
        
def test_stratified_sample_missing_groups_in_cases():
    """A sub-group is missing in controls - should be omitted from the result.
    """
    sizes = pd.Series({'male': 50, 'female': 60, 'dog': 15})
    cases, ctls = make_test_cohorts(sizes, ctl_scales=2)
    # remove dogs from the cases
    cases = cases.query('fact0 != "dog"')
    
    # expect controls to have 2x males and females, but no dogs at all
    for sf in [2, None]:  # expect the same results for None and 2
        smp = stratified_sample(ctls, cases, ['fact0'], size_factor=2)  
        assert_equal_proportions(smp)
        assert_group_sizes(smp.query('value == 1'), {'male': 50, 'female': 60})
        assert_group_sizes(smp.query('value == 0'), {'male': 2 * 50, 'female': 2 * 60})  


def test_ensure_gender_proportions__no_resampling_needed():
    """ A configuration with the same gender proportions in cases and controls
    should result in no resampling.
    """
    tmp = np.random.randint(0, 2, size=100)
    cohort = pd.Series(np.concatenate(2 * [tmp]))
    gender = pd.Series(['m', 'f'] * len(tmp))
    
    output = ensure_gender_proportions(cohort, gender)
    assert cohort.equals(output)
    
    # alternatively, with only one gender no resampling is needed too
    for gnd in ['male', 'female']:
        gender = pd.Series([gnd] * len(cohort))
        output = ensure_gender_proportions(cohort, gender)
        assert cohort.equals(output)
    
    
def test_ensure_gender_proportions__bad_inputs():
    """ Checks for exceptions thrown for bad inputs.
    """
    with pytest.raises(ValueError):
        for val in (0, 1):
            # no cases/controls
            ensure_gender_proportions(
                pd.Series([val] * 10),
                pd.Series(['female', 'male'] * 5))
            
    with pytest.raises(ValueError):
        # incorrect number of genders
        ensure_gender_proportions(
            pd.Series([0, 1] * 5),
            pd.Series(['female', 'male'] * 4 + ['dog'] * 2))
        
    with pytest.raises(ValueError):
        # no gender for some subjects
        cohort = pd.Series([0, 1] * 5)
        ensure_gender_proportions(
            cohort,
            pd.Series(['female', 'male'] * 4)).reindex(cohort.index)


def test_ensure_gender_proportions():
    """Tests whether the resampling actually happens, so when the input gender
    proportions differ.
    """
    # cases have more females so after resampling expect equal counts 
    ix = range(1234, 1234 + 600)
    cohort = pd.Series([1] * 200 + [0] * 400, index=ix)
    gender = pd.Series(['m', 'f'] * 100 + ['m'] * 100 + ['f'] * 300, index=ix)
    output = ensure_gender_proportions(cohort, gender)
    
    assert len(output) == 400
    assert output.value_counts().to_list() == [200, 200]  # cases/controls
    # equal gender proportions
    assert gender.loc[output.index].value_counts().to_list() == [200, 200]
    
    # extra consistency: cases remain cases
    assert cohort.reindex(output.index).equals(output)


def test_ensure_gender_proportions_resampling_of_larger_group():
    """Makes sure it is the larger group (cases/controls) that gets 
    resampled in order to minimize overall sample loss
    """
    # there are significantly more cases than controls (a real use case)
    cohort = pd.Series([1] * 382308 + [0] * 12381)
    gender = pd.Series(['m'] * 207740 + ['f'] * 174568  # cases
                       + ['m'] * 5197 + ['f'] * 7184)   # controls
    output = ensure_gender_proportions(cohort, gender)\
        .pipe(lambda s: pd.concat([s, gender], axis=1, keys=['value', 'gender']))\
        .dropna()
    
    assert_equal_proportions(output)
    assert len(output) == 313234
    assert output['value'].value_counts().to_list() == [300853, 12381]  # cases/controls
    
    # extra consistency: cases remain cases
    np.equal(cohort.reindex(output.index), output['value'].astype('int'))
