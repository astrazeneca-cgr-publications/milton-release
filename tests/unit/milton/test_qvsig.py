import pytest
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import scipy.sparse as sp

from milton.qvsig import (
    fast_fisher, 
    calc_fisher_oddsr_ci, 
    qv_significance)


def equal_scalars(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    return a == b


def test_fast_fisher():
    # the following data defines 3 contingency tables, which are 
    # inputs to the scipy's fisher_exact() funciton
    # 
    # The three tables test the following:
    # 1. inputs produce a valid result
    # 2. some counts are zero, oddsratio becomes np.nan, p-value is 1.0
    # 3. table[1, 0] * table[0, 1] == 0, oddsratio becomes inf
    ctables = np.array([
        [1, 3, 333, 331],
        [0, 0, 334, 334],
        [1, 0, 333, 334]
    ])
    
    out_oddsratio, out_pvalue = fast_fisher(ctables[:, 0], ctables[:, 1], 
                                            ctables[:, 2], ctables[:, 3])
    expected_oddsratio = []
    expected_pvalue = []
    
    for i in range(len(ctables)):
        ct = ctables[i].reshape((2, 2))
        oddsr, pval = fisher_exact(ct, alternative='two-sided')
        expected_oddsratio.append(oddsr if np.isfinite(oddsr) else np.nan)
        expected_pvalue.append(pval)
        
    assert len(out_oddsratio) == 3
    assert len(out_pvalue) == 3
    assert all(equal_scalars(a, b) for a, b in zip(out_oddsratio, expected_oddsratio))
    assert np.allclose(out_pvalue, expected_pvalue)


def test_calc_fisher_oddsr_ci():
    ctables = np.array([
        [10, 12, 35, 40],  # non-zero, finite odds ratio
        [0, 5, 10, 10],    # odds ratio == 2
        [10, 15, 0, 0],    # odds ratio is inf
    ])
    oddsratios = np.array([fisher_exact(ctables[i].reshape((2, 2)))[0]
                           for i in range(len(ctables))])

    ci_info = calc_fisher_oddsr_ci(oddsratios, 
                                   ctables[:, 0], ctables[:, 1], 
                                   ctables[:, 2], ctables[:, 3])
    oddsr, ci_lower, ci_upper = ci_info

    assert oddsr.shape == oddsratios.shape
    assert ci_lower.shape == ci_upper.shape
    assert ci_lower.shape == oddsr.shape
    assert 0 < ci_lower[0] < ci_upper[0]
    assert ci_lower[0] < oddsr[0] < ci_upper[0]

    # degenerate cases
    assert np.isfinite(oddsr).tolist() == [True, False, False]
    for ci in [ci_lower, ci_upper]:
        assert np.all(np.isnan(ci[1:]))


@pytest.fixture
def fake_genotype_matrix():
    ix = ['patient ' + n for n in 'ABCDE']
    cols = [f'gene {n}' for n in '123']
    mx = sp.csr_matrix(np.array([
        [1, 0, 1],
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]))
    return ix, cols, mx


def test_qv_significance(fake_genotype_matrix):
    ix, cols, mx = fake_genotype_matrix
    cohort = pd.Series([True, True, False, False], index=ix[:4])  # one less
    result = qv_significance(cohort, ix, cols, mx)

    assert len(result) == len(cols)  # genes
    assert (result['nSamples'] == 4).all()
    assert (result['pValue'] <= 1).all()

    counts = result[['BinQVcases', 'BinCaseFreq',
                     'BinQVcontrols', 'BinCtrlFreq']].values

    expected = np.array([
        [2, 1.0, 0, 0.0],
        [1, 0.5, 0, 0.0],
        [2, 1.0, 1, 0.5]
    ])

    assert np.all(counts == expected)

    # with such small cohort size odds ratio and its CI are all undefinded
    oddsr = result[['BinOddsRatio', 'BinOddsRatioLCI', 'BinOddsRatioUCI']]
    assert oddsr.isna().all().all()
