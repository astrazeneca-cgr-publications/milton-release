import numpy as np
import pandas as pd
import pytest

from milton.processing import *
from milton.data_desc import Col


def row_freq(df, by_col):
    return df.groupby(by_col).size().div(len(df))


def equal_freq(df1, df2, by):
    return np.allclose(row_freq(df1, by), row_freq(df2, by), rtol=0.01)

    
def test_poststratify():
    nums = np.arange(1, 11)
    df = pd.DataFrame({
        'a': [1] * 10000, 
        'b': np.random.choice(nums, 10000, replace=True, p=nums/nums.sum())})
    
    # equal frequency of all numbers, but shorter
    ref_df = pd.DataFrame({
        'b': list(nums) * 50
    })
    
    result = poststratify(df, 'b', ref_df, same_size=False)
    assert equal_freq(result, ref_df, 'b') 
    assert len(result) > len(ref_df)
    
    result = poststratify(df, 'b', ref_df, same_size=True)
    assert equal_freq(result, ref_df, 'b') 
    assert len(result) == len(ref_df)
    

def test_poststratify_with_scores():
    src = pd.DataFrame({
        # 2x more females as males
        'gender': ['M'] * 3 + ['F'] * 7
    })
    # data to be resampled:
    dst = pd.DataFrame({
        # 3x more males as females
        'gender': ['M'] * 10 + ['F'] * 3,
        'score': np.arange(13)
    })
    out = poststratify(dst, 'gender', src, scores=dst['score'])
    # the result should keep all females and drop most males
    # all males should have the lowest scores values
    expected = pd.DataFrame({
        'gender': ['F', 'F', 'F', 'M'],
        'score': [
            10, 11, 12,  # all females
            0,  # best male
            ]
    }, index=[10, 11, 12, 0])  # index values are the same as scores
    pd.testing.assert_frame_equal(out.sort_values(['gender', 'score']), 
                                  expected)
    
    
    
def test_resample():
    nums = np.arange(1, 11)
    df0 = pd.DataFrame({
        'a': [1] * 10000, 
        'b': np.random.choice(nums, 10000, replace=True, p=nums/nums.sum())})
    
    # equal frequency of all numbers, but shorter
    df1 = pd.DataFrame({
        'a': 1,
        'b': list(nums) * 50
    })
    
    df = pd.concat([df0, df1]).reset_index(drop=True)
    targets = pd.Series([0] * len(df0) + [1] * len(df1), index=df.index)

    result = resample(df, targets, stratify_by=None, same_size=False)
    assert df.equals(result.reindex_like(df))
    
    result = resample(df, targets, stratify_by=None, same_size=True)
    tgt = targets.reindex_like(result)
    assert len(result)
    assert (row_freq(result, tgt) == .5).all()
    assert not equal_freq(result[tgt == 0], result[tgt == 1], 'b')
    
    result = resample(df, targets, stratify_by='b', same_size=False)
    tgt = targets.reindex_like(result)
    assert len(result)
    assert row_freq(result, tgt).nunique() == 2  # different class feqs - no equality
    assert equal_freq(result[tgt == 0], result[tgt == 1], 'b')
    
    result = resample(df, targets, stratify_by='b', same_size=True)
    tgt = targets.reindex_like(result)
    assert len(result)
    assert (row_freq(result, tgt) == .5).all()
    assert equal_freq(result[tgt == 0], result[tgt == 1], 'b')
    

def test_drop_correlated():
    x = pd.DataFrame({
        'a': 2 * np.linspace(0, 10, 100) + .1 * np.random.randn(100),
        'b': 4 * np.linspace(0, 10, 100) + .1 * np.random.randn(100),
        'c': np.random.uniform(size=100),
        'd': np.random.uniform(size=100),
    })

    dt = DataTransformer(drop_correlated=.5)
    y = pd.Series([0] * len(x))
    result = dt.fit_transform(x, y)
    
    assert result.shape == (100, 3)
    assert 'a' in result or 'b' in result
    assert 'c' in result and 'd' in result
    
    
@pytest.fixture
def data():
    a = np.arange(0, 10).astype('float')
    
    data = pd.DataFrame({
        Col.AGE: np.arange(35, 85, 5),
        Col.GENDER: pd.Categorical(['Male', 'Female'] * 5),
        'a': a,
        'b': np.where(a < 5, np.nan, a),
        'c': np.where(a > 7, np.nan, a),
    })
    return data


@pytest.fixture
def targets():
    return pd.Series([0, 1] * 5)


def test_datatransformer_no_na_impute(data, targets):
    tfmr = DataTransformer(
        na_imputation=False,
        drop_cols=[Col.AGE, Col.GENDER])
    df = tfmr.fit_transform(data, targets)
    
    assert Col.AGE not in df
    assert Col.GENDER not in df
    assert len(df) == len(data)
    
    data_val = data.drop([Col.AGE, Col.GENDER], axis=1)
    assert (data_val.isna().sum() == df.isna().sum()).all()
    

def test_datatransformer_imputes_median(data, targets):
    tfmr = DataTransformer(
        na_imputation='median',
        scaling=None)
    df = tfmr.fit_transform(data, targets)
    
    for col in ['b', 'c']:
        assert (df[col] == data[col].fillna(data[col].median())).all()


def test_gender_specific_na_imputer():
    df0 = pd.DataFrame({
        Col.GENDER: ['Male'] * 5 + ['Female'] * 5,
        'A': [2, 3, 4, None, None] + [None, None, 4, 5, 6],
        'B': [5] * 5 + [3] * 5
    })

    # invert the order of gender in test data
    df1 = pd.DataFrame({
        Col.GENDER: ['Female'] * 5 + ['Male'] * 5,
        'A': [np.nan] * 10,
        'B': [np.nan] * 10
    }, index=[5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    
    im = GenderSpecImputer(GenderSpecNAStrategy(males='mean', females=17.0))
    im.fit(df0[['A', 'B']], df0[Col.GENDER])
    
    result = pd.DataFrame(
        im.transform(df1[['A', 'B']], df1[Col.GENDER]),
        index=df1.index, columns=['A', 'B'])
    
    expected = pd.DataFrame({
        'A': [17.0] * 5 + [3.0] * 5,
        'B': [17.0] * 5 + [5] * 5
    }, index=df1.index)
    
    pd.testing.assert_frame_equal(result, expected)
    
    # test the same but as part of data transformer
    dt = DataTransformer(
        scaling=None,
        na_imputation='median', 
        na_imputation_extra={
            'A': GenderSpecNAStrategy(males=-10, females=-20)
        })
    dt.fit(df0)
    result = dt.transform(df1)
    
    expected = pd.DataFrame({
        Col.GENDER: df1[Col.GENDER],
        'A': [-20.0] * 5 + [-10.0] * 5,
        'B': df0['B'].median(),
    }, index=df1.index)
    pd.testing.assert_frame_equal(result, expected)


def test_categorical_na_imputation():
    df = pd.DataFrame({
        'A': pd.Series(['a', pd.NA, 'c']).astype('category'),
        'B': pd.Series(['a', pd.NA, 'c']).astype('category')
    })
    dt = DataTransformer(
        scaling=None,
        na_imputation='median', 
        na_imputation_extra={
            'A': CategoricalNAStrategy()
            # column B will not produce dedicated NAN column
        })
    out = dt.fit_transform(df)
    expected = pd.DataFrame({
        'A::a':   [1, 0, 0],
        'A::c':   [0, 0, 1],
        'A::nan': [0, 1, 0],
        'B::a':   [1, 0, 0],
        'B::c':   [0, 0, 1]
    }, dtype='float')
    pd.testing.assert_frame_equal(expected, out.sort_index(axis=1))
