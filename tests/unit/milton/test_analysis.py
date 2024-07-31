"""Tests the whole pipeline on dummy data.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from milton import *

def test_basic_pipeline(tmp_path):
    """Testing the full pipeline on dummy datasets.
    """
    root = Path(tmp_path)
    sess = Session('local')
    conf = Settings()
    conf().analysis.min_cases = 10
    conf().analysis.min_prediction_thresh = .5 
    conf().features.biomarkers = True
    conf().patients.spec = pd.Series(1, index=range(1, 51))

    # fit models, estimates feature importance
    result = sess.evaluate(conf, debug=True)

    # collapsing analysis and rest of pipeline
    result.save_report(root)
    assert (root / 'metrics.csv').exists()
    assert (root / 'model_coeffs.csv').exists()
    assert (root / 'qv_significance.parquet').exists()
    assert (root / 'report.html').exists()
    assert (root / 'scores.parquet').exists()
    assert (root / 'stuff.pickle').exists()
    sess.stop()


def test_pipeline_with_feature_selection(tmp_path):
    root = Path(tmp_path)
    sess = Session('local')
    conf = Settings()
    conf().analysis.min_cases = 10
    conf().analysis.min_prediction_thresh = .5   
    conf().analysis.model_type = ModelType.STANDARD  
    conf().features.biomarkers = True
    conf().features.olink = True
    conf().feature_selection.iterations = 1
    conf().patients.spec = pd.Series(1, index=range(1, 51))

    # fit models, estimates feature importance
    result = sess.evaluate(conf, debug=True)

    # collapsing analysis and rest of pipeline
    result.save_report(root)
    assert (root / 'metrics.csv').exists()
    assert (root / 'model_coeffs.csv').exists()
    assert (root / 'qv_significance.parquet').exists()
    assert (root / 'report.html').exists()
    assert (root / 'scores.parquet').exists()
    assert (root / 'stuff.pickle').exists()
    sess.stop()
