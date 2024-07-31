import pytest
import dask
from pathlib import Path
import pandas as pd
from collections import namedtuple
from milton.data_source import ParquetDataSet, UkbDataStore, UkbFeatureBuilder
from milton.data_desc import Fld
from milton import data_source


@pytest.fixture
def fake_derived_resources(monkeypatch):
    """Fakes reading of static resources in derived biomarkers.
    """
    def read_csv(*args, **kwargs):
        return pd.DataFrame({'eid': [1, 2, 3]})
    monkeypatch.setattr(pd, 'read_csv', read_csv)


@pytest.fixture
def dask_local():
    # sets up a single-threaded scheduler so that dask tasks
    # can be executed in local context
    with dask.config.set(scheduler='single-threaded'):
        yield
        
        
@pytest.fixture
def install_parquet(monkeypatch):
    """Installs a dataframe under a dataset name so that this data is read by
    the parquet reading code. The data frame is split into a number of chunks
    to emulate multiple parquet partitions.
    """
    datasets = {}
    MARKER = '__TEST__'
    dtype_map = {
        # incomplete but enough for testing purposes
        'category': ParquetDataSet.STRING_TYPE,
        'float64': ParquetDataSet.FLOAT_TYPE,
        'int64': ParquetDataSet.INT_TYPE,
        'datetime64[ns]': ParquetDataSet.TS_TYPE,
    }
    
    def find_files(self, path):
        n = len(datasets[self.tag])
        # the special form of the path allows for matching tag and part id later
        return [(Path(path) / f'{MARKER}{self.tag}-{i}') for i in range(n)]
                    
    def load_schema(self):
        df = datasets[self.tag][0]
        dtypes = df.dtypes.astype('str')
        return {col: dtype_map[dtype] for col, dtype in dtypes.items()}
    
    def read_raw_file(self, path, columns):
        tag, n = str(path).split(MARKER)[1].split('-')
        return datasets[tag][int(n)][sorted(columns)]
                    
    monkeypatch.setattr(ParquetDataSet, '_find_files', find_files)
    monkeypatch.setattr(ParquetDataSet, '_load_schema', load_schema)
    monkeypatch.setattr(ParquetDataSet, '_read_raw_file', read_raw_file)
    
    def install_func(table, df, part_sizes=None):
        """Installs a dataframe under the tag and splits it into chunks
        of specified lengths.
        """
        nonlocal datasets
        n = len(df)
        assert n >= 2, 'Dataframes have to have at least 2 rows'
        if not part_sizes:
            chunks = [df.iloc[:n//2], df.iloc[n//2:]]
        else:
            chunks = []
            i = 0
            for ps in part_sizes:
                chunks.append(df.iloc[i:(i + ps)])
                i += ps
        datasets[table] = chunks
    
    return install_func


@pytest.fixture
def data_dict():
    """A mock of UkbDataDict with minimum functionality needed by some tests.
    """
    class TestDataDict:
        def __init__(self):
            self.feature_encodings = {
                int(Fld.ETHNICITY): {
                    '1': 'White',
                    '2': 'Mixed',
                    '3': 'Asian or Asian British',
                    '4': 'Black or Black British',
                    '5': 'Chinese',
                    '6': 'Other ethnic group'
                },
                int(Fld.GENDER): {'0': 'Female', '1': 'Male'},
                int(Fld.MONTH_OF_BIRTH): {
                    1: 'January',
                    2: 'February',
                    3: 'March',
                },
            }
            self.first_occurrence = {
                'N18': '10500',
                'N19': '10502',
                'M01': '10600',
                'M02': '10602',
            }
    return TestDataDict()


@pytest.fixture
def fake_datastore(monkeypatch, install_parquet, dask_local):
    """Returns a function that takes a dict(tag, dataframe) and sets up the
    contents of ALL datasets.
    """
    # predefined opt-out subjects
    monkeypatch.setattr(UkbDataStore, '_find_most_recent',
                        lambda self, table: Path(f'/test-path/{table}'))
    def installer(datasets, opt_outs=None):
        if opt_outs is None:
            opt_outs = pd.Index([])
        monkeypatch.setattr(data_source, '_load_opt_outs', 
                            lambda self, path: opt_outs )
        for table, df in datasets.items():
            # this patches the underlying parquet datasets
            install_parquet(table, df)
        return UkbDataStore(location='/some-fake-path', 
                            names=list(datasets),
                            opt_outs='some_file')
    return installer
