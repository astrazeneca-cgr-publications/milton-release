"""MILTON configuration constructor that was used to generate published results.
"""
import pandas as pd
from pathlib import Path
from . import ICD10
from .data_info import UKB_DATA_LOCATION
from .configuration import Settings, ModelType
from .data_desc import Col
from .processing import GenderSpecNAStrategy
from .globs import DD


def make_v16_config(*, 
                    ancestry, 
                    ctl_ratio=19, 
                    time_model=ModelType.STANDARD, 
                    feature_set='67bm', 
                    feature_selection_iterations=None):
    """Configuration generator for the different predefined scenarios.
    
    ancestry (obj): 
      Takes path to file containing relevant sample IDs
    ctrl_ratio (int): Ratio of controls to cases (also used to determine number 
      of iterations for olink feature selection). 19 for EUR ancestry and 
      67 biomarkers, 9 for all other ancestries/feature-sets
    time_model (obj): 
      Time-model to subset cases according to time lag between diagnosis and 
      sample collection. Takes one of the following {ModelType.STANDARD, 
      ModelType.PROGNOSTIC, ModelType.DIAGNOSTIC} for time-agnostic, prognostic 
      and diagnostic time-models, respectively.
    feature_set (str): 
      Pre-implemented feature set to run MILTON on. Takes one of {'67bm', 
      'olink-only', 'olink-and-67bm'}
    feature_selection_iterations: 
      Number of iterations to perform Boruta feature selection for. In each 
      iteration a different age-sex-number matched control set is sampled. Union
      of confirmed (or tentative, if selected) features across all iterations 
      are used for further training XGBoost classifier. 
        - Unless otherwise stated, ctl_ratio is used as number of iterations for
          feature_selection.
        - Set to 0 if no feature pre-selection is desired.
        - No feature selection is performed for 67 biomarkers.
        - To preserve any co-variates or features, use the 
          conf().feature_selection.preserved option.
    """
    
    conf = Settings()

    if ancestry:
        # use specific settings, not the defaults
        qv_model_dir, qv_subject_subset = ancestry
        conf().analysis.qv_model_dir = qv_model_dir
        conf().analysis.qv_subject_subset = qv_subject_subset
        # specific ancestry means training is also constrained
        ids = pd.read_csv(qv_subject_subset, usecols=['eid'])['eid']
        conf().patients.used_subjects = ids.to_list()
        
    if ctl_ratio:
        conf().patients.controls_to_cases_ratio = ctl_ratio
        
    if feature_selection_iterations is None:
        feature_selection_iterations=ctl_ratio
    
    if feature_set=='67bm': # 67 biomarkers only
        conf.features.biomarkers = True
        conf.features.respiratory=True
        conf.features.overall_health=True
        conf.features.olink = False
        conf.features.olink_covariates = False
    
    elif feature_set=='olink-only': # olink proteomics only
        conf.features.biomarkers = False
        conf.features.olink = True
        conf.features.olink_covariates = True

        conf().feature_selection.iterations = feature_selection_iterations
        conf().feature_selection.preserved = [  
            # all covariates
            Col.AGE, 
            Col.GENDER, 
            'Alcohol intake frequency.',
            'Illnesses of father',
            'Illnesses of mother',
            'Smoking status',
            'Blood-type haplotype',
            'Body mass index (BMI)'
        ]
    
    elif feature_set=='olink-and-67bm': # olink and 67 bm
        conf.features.biomarkers = True
        conf.features.respiratory=True
        conf.features.overall_health=True
        conf.features.olink = True
        conf.features.olink_covariates = True

        # number of iterations to do feature selection for
        conf().feature_selection.iterations = feature_selection_iterations 
        conf().feature_selection.preserved = [  
            # all covariates
            Col.AGE, 
            Col.GENDER, 
            'Alcohol intake frequency.',
            'Illnesses of father',
            'Illnesses of mother',
            'Smoking status',
            'Blood-type haplotype',
            'Body mass index (BMI)'
        ]
        # also don't do feature selection within 67 traits, just olink proteins
        ukb_biomarkers = DD.predefined(biomarkers=True, 
                                       respiratory=True, 
                                       overall_health=True)\
            .index.drop_duplicates()\
            .drop([Col.GENDER, Col.AGE])\
            .to_list()
        conf().feature_selection.preserved.extend(ukb_biomarkers)
    
    else:
        print('Feature set not defined. Proceeding with 67 biomarkers..')
        conf.features.biomarkers = True
        conf.features.respiratory=True
        conf.features.overall_health=True
        conf.features.olink = False
        conf.features.olink_covariates = False
        
    
    conf().preproc.na_imputation = 'median'
    conf().preproc.na_imputation_extra = {
        'Testosterone': GenderSpecNAStrategy(males='median', females='median'),
        Col.RHEUMATOID_FACTOR: ('constant', 0.0),
        Col.OESTRADIOL: GenderSpecNAStrategy(males=36.71, females=110.13),
    }

    conf().analysis.default_model = 'xgb'
    conf().analysis.hyper_parameters = {
        'n_estimators': [50, 100, 200, 300],
    }
    conf().analysis.hyper_param_metric = 'roc_auc' 
    
    # number of replicas to train XGBoost for. Different control sets 
    # (ctl_ratio x #cases) will be sampled per replica
    conf().analysis.n_replicas = 10  
    conf().analysis.evaluate_all_replicas = True
    
    conf().analysis.model_type = time_model
    return conf


def example_run():
    ctl_ratio=9
    time_model='time_agnostic'
    ancestry='EUR' # one of AFR, AMR, EAS, EUR, SAS 
    feature_set='olink-and-67bm'
    #specify which ICD10 code to sample cases and controls from
    code='I871'
    out_dir = Path('.') / 'results' / code / ancestry / feature_set / time_model
    out_dir.mkdir(exist_ok=True)

    if time_model=='time_agnostic':
        timemodel=ModelType.STANDARD
    elif time_model=='prognostic':
        timemodel=ModelType.PROGNOSTIC
    elif time_model=='diagnostic':
        timemodel=ModelType.DIAGNOSTIC
    else:
        print('time_model not defined')

    #call config function with relevant parameters
    settings = make_v16_config(
        ancestry= (Path(UKB_DATA_LOCATION) 
                   / 'sample_lists' 
                   / f'UKB470K_selected_{ancestry}'),
        ctl_ratio=ctl_ratio,
        time_model=timemodel,
        feature_set=feature_set)

    #specify which ICD10 code to use for cases and controls
    settings().patients.spec = ICD10.find_by_code(code)
    settings().analysis.min_cases = 0
