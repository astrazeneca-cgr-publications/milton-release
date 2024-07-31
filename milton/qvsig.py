from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
import numba
import math


# a large cache of log-factorials
_N_LOG_FACT = 600000
_LOG_FACTORIALS = np.zeros(_N_LOG_FACT)
_LOG_FACTORIALS[1:] = np.log(np.arange(1, _N_LOG_FACT)).cumsum()


@numba.vectorize
def hypergeom_probability(x, n, K, N):
    """Optimized hypergeometric probability distribution. It calculates the 
    standard formula, which is based on factorials
    (https://en.wikipedia.org/wiki/Hypergeometric_distribution). The key aspect 
    of the optimization is precomputation of large-enough number of factorials
    and vectorization of the code.
    """
    def lnfactorial(n):
        """Logarithm of n! with algorithmic approximation
        """
        return _LOG_FACTORIALS[n] if n < _N_LOG_FACT else math.lgamma(n + 1)
    
    def lncombination(n, p):
        """Logarithm of the number of combinations of 'n' objects taken 
        'p' at a time.
        """
        return lnfactorial(n) - lnfactorial(p) - lnfactorial(n - p)
    
    return math.exp(lncombination(K, x) 
                    + lncombination(N - K, n - x) 
                    - lncombination(N, n))


@numba.njit(parallel=True, nogil=True)
def _fast_fisher(a_true, b_true, a_false, b_false):
    M = len(a_true)  # number of tests to perform
    pvalues = np.full(M, 1.0, np.float64)
    oddsratios = np.empty(M, np.float64)
    
    for i in numba.prange(M):
        k = a_true[i]  # number of successes
        n = a_true[i] + a_false[i]   # number of draws (cohort size)
        K = a_true[i] + b_true[i]
        N = K + a_false[i] + b_false[i]
        low_x = max(0, n - (N - K))
        high_x = min(n, K)
        
        # odds ratios
        odds_denom = a_false[i] * b_true[i]
        if odds_denom > 0:
            oddsratios[i] = (a_true[i] * b_false[i]) / odds_denom
        else:
            oddsratios[i] = np.nan
            continue  # pvalue will be 1.0
        
        # to calculate the tail areas, integrate the entire domain of the 
        # hypergeometric distribution filtering out everything above the 
        # probability of the measured result (cutoff)
        if low_x != high_x:
            two_tail = 0.0
            cutoff = hypergeom_probability(k, n, K, N)
            for x in range(low_x, high_x + 1):
                p = hypergeom_probability(x, n, K, N)
                if p <= cutoff:
                    two_tail += p
            if two_tail < 1.0:
                pvalues[i] = two_tail
                
    return oddsratios, pvalues


def fast_fisher(a_true, b_true, a_false, b_false):
    """Fast 2-sided fisher test. The function takes 4 numpy int arrays, of the
    same length, which contain the values of a number of contingency tables.
    Each table has the following layout:
        [[a_true, b_true],
         [a_false, b_false]]
    For example, As and Bs are cases/controls while true/false values represent
    subject counts that have a propert or don't (eg a phenotype).
    
    Returns
    -------
    oddsratios, pvalues - two float arrays of the same length as inputs
    """
    if a_true.ndim != 1:
        raise ValueError('Only 1-dim arrays are accepted.')
    if any(ar.shape != a_true.shape for ar in (b_true, a_false, b_false)):
        raise ValueError('Arrays must have the same lengths.')
    if any(np.any(ar < 0) for ar in (a_true, b_true, a_false, b_false)):
        raise ValueError('Arrays must have non-negative values.')
    
    return _fast_fisher(np.asarray(a_true, np.int64),
                        np.asarray(b_true, np.int64),
                        np.asarray(a_false, np.int64),
                        np.asarray(b_false, np.int64))


def qv_significance(cohort, genotype_ix, genotype_cols, genotype_matrix, 
                    all_ctl=False,
                    alpha=0.05):
    """Calculate QV significance via fisher exact test, using the vectorized 
    implementation of the test and working with sparse genotype matrices.
    
    Parameters
    ----------
    cohort : pandas Series with patient ID in index, and 0/1 values indicating 
        controls and cases respectively.
    genotype_ix : patient IDs of corresponding to rows of the genotype matrix
    genotype_cols : column names of the genotype matrix
    genotype_matrix : sparse matrix with non-negative integers
    all_ctl : when True, ignore controls defined in the cohort and use all 
        controls available in the genotypes matrix,
    alpha : significance level for calulation of odds ratio CI
    
    Returns
    -------
        qv_significance : pandas DataFrame with fisher test results for 
            each gene. Columns are: values of test statistic, test p-values,
            number of cases and controls with the 
    """
    genotype_ix = pd.Index(genotype_ix)
    # cohort has to be shrunk down to the matrix' index (the matrix may not
    # have all of UK Biobank patients)
    cohort = cohort.reindex(genotype_ix).dropna()
    # extract matrix row numbers by patient ID
    cohort_ix = cohort.index.map(lambda v: genotype_ix.get_loc(v))
    
    genotypes_bin = genotype_matrix.sign()  # flatten values to 0, 1
    cases_ix = cohort_ix[cohort == 1]
    cases = genotypes_bin[cases_ix]

    if not all_ctl:
        controls = genotypes_bin[cohort_ix[cohort == 0]]
    else:
        # controls are everything but the cases
        ix = np.arange(genotypes_bin.shape[0])
        controls = genotypes_bin[~np.isin(ix, cases_ix)]

    a = np.asarray(cases.sum(axis=0), np.int64).ravel()
    b = np.asarray(controls.sum(axis=0), np.int64).ravel()
    # with sparse matrices, zeros have to be counted differently
    c = cases.shape[0] - a
    d = controls.shape[0] - b
    
    oddsratio, pvalue = fast_fisher(a, b, c, d)
    ci_info = calc_fisher_oddsr_ci(oddsratio, a, b, c, d, alpha)
    oddsratio, ci_lower, ci_upper = ci_info
    
    return pd.DataFrame({
        'pValue': pvalue,
        'nSamples': cases.shape[0] + controls.shape[0],
        'BinQVcases': a,
        'BinQVcontrols': b,
        'BinCaseFreq': a / cases.shape[0],
        'BinCtrlFreq': b / controls.shape[0],
        'BinOddsRatio': oddsratio,
        'BinOddsRatioLCI': ci_lower,
        'BinOddsRatioUCI': ci_upper,
        
    }, index=pd.Index(genotype_cols, name='Gene'))


def calc_fisher_oddsr_ci(oddsratio, a_true, b_true, a_false, b_false, 
                         alpha=0.05):
    cont_tables = np.vstack([a_true, b_true, a_false, b_false]).reshape((2, 2, -1))
    cont_tables = np.where(cont_tables == 0, np.nan, cont_tables)
    oddsratio_se = np.sqrt(1 / cont_tables).sum(axis=(0, 1))
    offs = norm.ppf(1 - alpha / 2, loc=0, scale=1) * oddsratio_se
    clean_oddsratio = np.where(np.isfinite(oddsratio) & (oddsratio > 0),
                               oddsratio, np.nan)
    logodds = np.log(clean_oddsratio)

    return clean_oddsratio, np.exp(logodds - offs), np.exp(logodds + offs)


class CollapsingAnalyser:
    """Utility that loads/caches QV model data and performs collapsing analysis.
    """
    def __init__(self, mdl_loc):
        self.loc = Path(mdl_loc)
        paths = [p for p in self.loc.iterdir() if p.name.endswith('.pickle')]
        # a crude way of extracting QV model name
        def get_mdl_name(path):
            parts = path.name.split('.')[0].split('_')
            # last word before "matrix"
            return parts[parts.index('matrix') - 1]
        
        self.mdl_names = [get_mdl_name(p) for p in paths]
        self._paths = dict(zip(self.mdl_names, paths))
        self._models = {}
        
    def get_model(self, name):
        if name not in self._models:
            with self._paths[name].open('rb') as f:
                self._models[name] = pickle.load(f)
        return self._models[name]
    
    def __call__(self, mdl_name, cohort, pval_thresh=None, all_ctl=False):
        result =  qv_significance(cohort, 
                                  *self.get_model(mdl_name), 
                                  all_ctl=all_ctl)
        if pval_thresh:
            return result[result['pValue'] <= pval_thresh]
        else:
            return result
