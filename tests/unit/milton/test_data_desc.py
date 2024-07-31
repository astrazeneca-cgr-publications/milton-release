import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from milton.data_desc import biomarker, Col, Fld, DerivedBiomarkers


@pytest.fixture
def input_df():
    """A dict simulating an input data frame for biomarker calculation.
    """
    class FakeDF:
        def __init__(self, n):
            self.n = n
            self.data = {}
            
        def __len__(self):
            return self.n
        
        def __iter__(self):
            return iter(self.data)
        
        def __getitem__(self, column):
            if column not in self.data:
                if column == Fld.GENDER:
                    values = pd.Series(['Male', 'Female'] * (self.n // 2), 
                                       dtype='category')
                elif column == Fld.ETHNICITY:
                    values = pd.Series(['1001', '3001'] * (self.n // 2), 
                                       dtype='category')
                elif column == Fld.MONTH_OF_BIRTH:
                    values = pd.Series(['January', 'May'] * (self.n // 2),
                                       dtype='category')
                elif column == Fld.YEAR_OF_BIRTH: 
                    values = pd.Series([1978, 2018] * (self.n // 2))
                else:
                    values = pd.Series([1] * self.n, dtype='float')
                self.data[column] = values
            return self.data[column]
        
    return FakeDF(6)


def test_biomarker_decorator():
    class MyBiomarkers:
        dependencies = {}

        @biomarker(dependencies, 'a', 'b', 'c')
        def marker_0(self):
            return 0

        @biomarker(dependencies, 'x', 'x', 'x', 'y')
        def marker_1(self):
            return 1

        @biomarker(dependencies, 'a', 'x')
        def marker_2(self):
            return 2

    mb = MyBiomarkers()

    expected = {
        'marker_0': ['a', 'b', 'c'],
        'marker_1': ['x', 'y'],
        'marker_2': ['a', 'x']
    }
    assert MyBiomarkers.dependencies == expected
    assert mb.marker_0() == 0
    assert mb.marker_1() == 1
    assert mb.marker_2() == 2


@pytest.mark.parametrize('biomarker', DerivedBiomarkers.DEPENDENCIES)
def test_biomarkers_by_name(fake_derived_resources, input_df, biomarker):
    bm = DerivedBiomarkers('some-location', [biomarker])
    assert bm.names == [biomarker]

    res = bm.calculate(input_df, names=[biomarker])
    assert res.columns.to_list() == [biomarker]
    assert len(res) == 6  # as in input_df
    # keys referenced in bm.calculate() are recorded, but correct for the BMI
    # hack - dependencies require 2 BMI versions, feature builder fuses them
    # into a single Fld.BMI instance
    assert set(input_df) == set(bm.deps_of(biomarker)).difference({Fld.BMI_v2})
    assert res[biomarker].notna().all()
    
    
def test_biomarkers_calc_all(fake_derived_resources, input_df):
    bm = DerivedBiomarkers('some-location', True)
    assert sorted(bm.names) == sorted(DerivedBiomarkers.DEPENDENCIES)
    res = bm.calculate(input_df)
    assert set(input_df) == set(bm.dependencies).difference({Fld.BMI_v2})  # BMI hack - see test_biomarkers_by_name()
    assert len(res) == 6
    assert sorted(res.columns) == sorted(bm.names)
    assert res.notna().all().all()
