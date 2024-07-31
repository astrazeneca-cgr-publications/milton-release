from milton.data_info import DATASETS, HESIN_DIAG
import pytest
import numpy as np
import pandas as pd

from milton.data_source import (
    DataSourceException,
    ParquetDataSet,
    UkbDataset,
    UKB,
)
from milton import data_source


# predefined opt-out subjects
OPT_OUTS = [13, 113]


def test_testing_infrastructure(dask_local, install_parquet, fake_datastore):
    # an echo test: check if you get the same data back
    df_ukb = pd.DataFrame({
        'eid': [1, 2, 3, 4, 5],
        '30080-0.0': [308.0, 311.0, 252.0, 238.2, 370.0],
        '30080-1.0': [250.0, 362.0, 229.4, 235.0, 236.0],
    })
    install_parquet(UKB, df_ukb)
    
    ds = UkbDataset()
    df = ds.read_data(pd.Series(['30080-0.0', '30080-1.0']))\
        .concat().compute()
    pd.testing.assert_frame_equal(df_ukb.set_index('eid'), df)
    
    # now the datasource - the same thing
    dst = fake_datastore({UKB: df_ukb})
    df = dst[UKB].load(['30080-0.0', '30080-1.0']).concat().compute()
    pd.testing.assert_frame_equal(df_ukb.set_index('eid'), df)
    
    
def test_table_column_names(dask_local, fake_datastore):
    tall_df = pd.DataFrame({
        'eid': list(range(5)) + list(range(5)),
        'A': np.arange(10),
        'B': [5] * 10
    })
    dst = fake_datastore({ds: tall_df for ds in DATASETS} | {
        UKB: pd.DataFrame({
            'eid': np.arange(5),
            '123-0.0': np.arange(5),
            '123-0.1': np.arange(5),
            '200-0.0': np.arange(5).astype('float'),
            '200-1.0': np.arange(5).astype('float'),
            '200-1.1': np.arange(5).astype('float')
            }),
    })
    
    def get_schema(table, fields):
        # an abbreviation
        return list(dst[table].get_schema(fields))
    
    assert get_schema(UKB, ['123']) == ['123-0.0', '123-0.1']
    assert get_schema(UKB, ['200']) == ['200-0.0', '200-1.0', '200-1.1']
    assert get_schema(UKB, ['123', '200']) == ['123-0.0', '123-0.1', 
                                               '200-0.0', '200-1.0', '200-1.1']
    assert get_schema(UKB, ['eid']) == ['eid']
    for table in DATASETS:
        if table != UKB:
            assert get_schema(table, ['A']) == ['A']
            assert get_schema(table, ['A', 'B']) == ['A', 'B']
            
    # check correct reporting of datatypes
    list(dst[UKB].get_schema(['123']).values()) == [ParquetDataSet.INT_TYPE] * 2
    list(dst[UKB].get_schema(['200']).values()) == [ParquetDataSet.FLOAT_TYPE] * 2
            
    # test if requests to non-existent fields get reported
    with pytest.raises(DataSourceException):
        get_schema(UKB, ['unknown-field'])
        
    with pytest.raises(DataSourceException):
        get_schema(HESIN_DIAG, ['unknown-field'])
            
            
def test_datastore_excludes_opt_outs(dask_local, fake_datastore):
    dst = fake_datastore({
        name: pd.DataFrame({
            'eid': list(range(8)) + OPT_OUTS,
            'A': [1] * 10,
            'B': [2] * 10
        })
        for name in DATASETS
    }, opt_outs=pd.Index(OPT_OUTS))  # predefined opt-out subjects
    for name in DATASETS:
        ix = dst[name].dataset.index
        assert ix.intersection(pd.Index(OPT_OUTS)).empty


# @pytest.fixture
# def feature_encodings():
#     # minimum set of feature encodings for tests
#     return {
#         Col.GENDER: pd.DataFrame({'value': [0, 1], 'meaning': ['Female', 'Male']}),
#         Col.ETHNICITY: pd.DataFrame({'value': [1001], 'meaning': 'British'})
#     }


# def test_parquet_dataset(fake_parquet_files, fake_dask_client):
#     df = pd.DataFrame({
#         'eid': np.arange(10, 110, 10),
#         'A': np.arange(1, 11),
#         'B': np.linspace(0, 1, 10)
#     })
#     fake_parquet_files({
#         'UKB-data': [df, 2 * df, 3 * df]
#     })
    
#     # a simple test that verifies that files are read in chunks and 
#     # map-gather correctly applies a function to each chunk
#     result = load_parquet('UKB-data', columns=['A'], index='eid')\
#         .map(lambda df: df.sum())\
#         .concat()\
#         .compute()
#     assert result.shape == (3,)
#     assert result['A'].to_list() == [55, 2*55, 3*55]


# def test_feature_builder(ukb_data, feature_encodings):
#     df = ukb_data()
#     df_out = UkbFeatureBuilder(feature_encodings).process(df)
    
#     # there should be no duplicated non-numeric columns
#     expected_cols = df.columns.drop_duplicates()
#     for c in df_out:
#         assert c in expected_cols, c
#         if c != 'Diagnoses - ICD10':
#             assert df_out[c].ndim == 1
#         else:
#             assert df_out[c].shape[1] == 2
    
#     assert df_out[Col.GENDER].dtype == 'category'
#     assert df_out[Col.ETHNICITY].dtype == 'category'
