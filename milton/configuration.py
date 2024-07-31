from dataclasses import dataclass, asdict, fields, replace, field
from typing import Optional, List, Tuple, Union, Sequence, Dict
from enum import Enum
from functools import cached_property
from pathlib import Path
import pandas as pd

from .data_source import UKB_DATA_LOCATION
from .data_desc import Col


class BaseConf:
    """Pretty printing of all configuration items.
    """
    
    def asdict(self):
        """Converts the entire configuration into a (nested)
        python dict.
        """
        return asdict(self)
    
    def __repr__(self):
        lines = [self.__class__.__name__ + ':']
        
        for f in fields(self):
            field_lines = str(getattr(self, f.name)).split('\n')
            line = field_lines[0]
            if len(line) > 120:
                line = line[:120 - 3] + '...'
            out_line = f'  {f.name} = {line}'
            lines.append(out_line)
            for ln in field_lines[1:]:
                lines.append('  ' + ln)
        return '\n'.join(lines)
    
    
@dataclass(repr=False)
class ClusterConf(BaseConf):
    # used only in the local run, batch mode conf is in batch.py
    kind: str = 'slurm'
    n_workers: int = 8 
    memory: str = '8G' 
    cores: int = 8
    port: Optional[int] = None
    silence_logs: str = 'warning'
    

@dataclass(repr=False)
class DatasetConf(BaseConf):
    location: Optional[str] = UKB_DATA_LOCATION
    

@dataclass(repr=False)
class PatientConf(BaseConf):
    training_controls: Optional[Sequence[int]] = None
    collapsing_controls: Optional[Sequence[int]] = None
    controls_to_cases_ratio: int = 4
    sampling_factors: Optional[List[Tuple[str, int]]] = ((Col.AGE, 4), 
                                                         (Col.GENDER, 0))
    infrequent_gender_thresh : float = 0.1
    diag_fields: Tuple[int] = (41270, 40001, 40002, 40006)
    spec: Optional[Union[str, List[str], pd.Series]] = None
    used_subjects: Optional[Sequence[int]] = None
    drop_same_chapter_controls: bool = True
    
    
@dataclass(repr=False)
class FeatureConf(BaseConf):
    biomarkers: bool = False
    respiratory: bool = False
    lifestyle: bool = False
    med_hist: bool = False
    overall_health: bool = False
    pulse_wave: bool = False
    mental_health: bool = False
    environmental: bool = False
    social: bool = False
    olink: Union[List[str], bool] = False
    olink_covariates: bool = False
    ukb_custom: List[int] = tuple()


@dataclass(repr=False)
class FtSelectionConf(BaseConf):
    # number of iterations, 0 means no feature selection
    iterations: int = 0  
    # add tentative features in addition to confirmed
    tentative: bool = False  
    # fraction of data to run it on
    data_frac: float = .5
    # features that always have to be selected
    preserved: List[str] = (Col.AGE, Col.GENDER)

    
@dataclass(repr=False)
class PreprocConf(BaseConf):
    drop_cols: Optional[List[str]] = None
    drop_na_frac: float = 1.0
    na_imputation: Optional[str] = 'median'
    na_imputation_extra: Optional[Dict] = None
    scaling: Optional['str'] = 'standard'
    dont_drop: Optional[List[str]] = None
    

class ModelType(Enum):
    STANDARD = 1
    PROGNOSTIC = 2
    DIAGNOSTIC = 3
    
    
@dataclass(repr=False)
class AnalysisConf(BaseConf):
    """Configuration of Analysis
    """
    min_cases: int = 100  # threshold for rejecting model fitting
    derived_biomarkers: Optional[str] = None
    min_prediction_thresh: float = .7
    ncr_quantiles: Tuple[float] = (.2, .3, .5, 1.)
    max_ncr: float = 10
    gene_sign_thresh: Optional[float] = 0.05
    n_replicas: int = 1   # trains 1 model instance by default
    evaluate_all_replicas: bool = True  # if False, only 1 replica is evaluated
    held_out_frac: Optional[float] = None  
    grid_search: bool = True
    correlation_thresh: float = .5
    qv_model_dir: str = 'qv_models/UKBWES/EUR'
    qv_subject_subset: Path = Path(UKB_DATA_LOCATION) / 'sample_lists/UKB470K_selected_EUR.txt'
    collapsing_on_data_index: bool = False
    disease_prevalence: Optional[float] = None
    default_model: str = 'xgb'
    hyper_parameters: Dict = field(default_factory=dict)
    hyper_param_metric: str = 'roc_auc'
    model_type: ModelType = ModelType.STANDARD
    # for the capped model type, string date of cutoff
    diagnosis_cap: Optional[str] = None
    # for prognostic/diagnostic models, this is the max accepted time difference
    # between diagnoses and measurements. Unit: years.
    max_time_lag: int = 10
    
    
@dataclass(repr=False)
class OutputConf(BaseConf):
    title: str = 'Milton Report'
    location: str = '.'
        
    model_performance: bool = True
    feature_importance: bool = True
    effect_sizes: bool = True
    embedding: bool = False
    feature_embeddings: Union[List[str], None, int] = None
        
    max_scatter_points: int = 2000
    cmap: str = 'coolwarm'
    aspect: float = 4/3


@dataclass(repr=False)
class Configuration(BaseConf):
    cluster: ClusterConf = None
    dataset: DatasetConf = None
    patients: PatientConf = None
    features: FeatureConf = None
    feature_selection: FtSelectionConf = None
    preproc: PreprocConf = None
    analysis: AnalysisConf = None
    output: OutputConf = None
    
    def __init__(self,
                 cluster=None, 
                 dataset=None,
                 patients=None,
                 features=None,
                 feature_selection=None,
                 preproc=None,
                 analysis=None,
                 output=None):
        self.cluster = cluster or ClusterConf()
        self.dataset = dataset or DatasetConf()
        self.patients = patients or PatientConf()
        self.features = features or FeatureConf()
        self.feature_selection = feature_selection or FtSelectionConf()
        self.preproc = preproc or PreprocConf()
        self.analysis = analysis or AnalysisConf()
        self.output = output or OutputConf()
        
    @staticmethod
    def read_yaml():
        raise NotImplementedError()

    @staticmethod
    def copy(conf):
        return Configuration(**{k: replace(v) for k, v in vars(conf).items()})
        
        
class SettingsGroup:
    def __init__(self, conf, name):
        self._conf = conf
        self._name = name
        
    def _properties(self):
        return sorted(name for name, val in type(self).__dict__.items()
                      if isinstance(val, property))
        
    def __repr__(self):
        lines = [self._name + ':']
        for setting in self._properties():
            value = getattr(self, setting)
            lines.append('  %s : %s' % (setting, value))
        return '\n'.join(lines)

        
class FeatureSettings(SettingsGroup):
    """Configuration of features that are extracted from UK Biobank.
    There are several groups of features that can be selected.
    Assign value True to feature groups you want to be included.
    """
    
    @property
    def biomarkers(self): 
        """Biomarker features include:
          - Assay biomarkers
          - Waist circumference
          - BMI
        """
        return self._conf.features.biomarkers
    
    @biomarkers.setter
    def biomarkers(self, value):
        self._conf.features.biomarkers = value
        
    @property
    def respiratory(self):
        """Respiratory features include:
          - Forced expiratory volume in 1-second (FEV1), Best measure
          - Forced vital capacity (FVC), Best measure
          - FEV1/ FVC ratio Z-score
        """
        return self._conf.features.respiratory
        
    @respiratory.setter
    def respiratory(self, value):
        self._conf.features.respiratory = value
    
    @property
    def lifestyle(self): 
        """Lifestyle features inlcude:
          - Alcohol intake frequency
          - Ever addicted to any substance or behaviour
          - Pack years of smoking
          - Sleep duration
          - Sleeplessness / insomnia
          - Snoring
          - Cannabis use frequency
        """
        return self._conf.features.lifestyle
    
    @lifestyle.setter
    def lifestyle(self, value):
        self._conf.features.lifestyle = value
    
    @property
    def medical_history(self):
        """Medical history features comprise:
          - virus seropositivity features
          - total number of operative procedures
          - end-stage renal disease report
        """
        return self._conf.features.med_hist
    
    @medical_history.setter
    def medical_history(self, value):
        self._conf.features.med_hist = value
        
    @property
    def overall_health(self):
        """Overall health features include:
          - Pulse rate, automated reading
          - Diastolic blood pressure, automated reading
          - Systolic blood pressure, automated reading
          - No-wear time bias adjusted average acceleration
          - Systolic blood pressure, manual reading
          - Diastolic blood pressure, manual reading"""
        return self._conf.features.overall_health
    
    @overall_health.setter
    def overall_health(self, value):
        self._conf.features.overall_health = value
        
    @property
    def pulse_wave(self):
        """Pulse wave features include:
          - Pulse wave Arterial Stiffness index
          - Pulse wave reflection index
          - Pulse wave peak to peak time
          - Position of the pulse wave peak
          - Position of pulse wave notch
          - Position of the shoulder on the pulse waveform
          - Absence of notch position in the pulse waveform
          - Cardiac index during PWA
        """
        return self._conf.features.pulse_wave
        
    @pulse_wave.setter
    def pulse_wave(self, value):
        self._conf.features.pulse_wave = value
        
    @property
    def mental_health(self): 
        """Mental health features include:
          - Worrier / anxious feelings
          - Ever had prolonged feelings of sadness or depression
          - Ever sought or received professional help for mental distress
          - Ever suffered mental distress preventing usual activities
          - Seen doctor (GP) for nerves, anxiety, tension or depression
          - Seen a psychiatrist for nerves, anxiety, tension or depression
        """
        return self._conf.features.mental_health
        
    @mental_health.setter
    def mental_health(self, value): 
        self._conf.features.mental_health = value
        
    @property
    def environmental(self): 
        """Environmental features include:
          - Nitrogen dioxide air pollution; 2010
          - Nitrogen oxides air pollution; 2010
          - Particulate matter air pollution (pm10); 2010
          - Particulate matter air pollution (pm2.5); 2010
          - Average 24-hour sound level of noise pollution
        """
        return self._conf.features.environmental
        
    @environmental.setter
    def environmental(self, value): 
        self._conf.features.environmental = value
        
    @property
    def social(self):
        """Social features include:
          - Qualifications
          - Current employment status - corrected
          - Ethnic background
          - Current employment status
        """
        return self._conf.features.social
        
    @social.setter
    def social(self, value): 
        self._conf.features.social = value
        
    @property
    def derived_biomarkers(self):
        """Derived biomarkers is a group of features computed from 
        data in UK Biobank. They include:
          - eGFR_EPI
          - eGFR_MDRD
          - UACR
          - ASTALT
          - FIB4
          - APRI
          - FLI
          - HSI
          - BARD
          - Malignant neoplasm survival times in years: 90 features 
              with the followng name pattern: ONC_Cxx, where Cxx is a 
              malignant neoplasm ICD10 code.
        Pass a list of names to include in your data or set to None.
        """
        return self._conf.analysis.derived_biomarkers
    
    @derived_biomarkers.setter
    def derived_biomarkers(self, value):
        self._conf.analysis.derived_biomarkers = value
        
    @property
    def olink(self):
        """OLINK features. Following values are accepted:
          - True: add all OLINK biomarkers
          - False: do not use OLINK data
          - List of strings: specific OLINK biomarkers to include.
        """
        return self._conf.features.olink
    
    @olink.setter
    def olink(self, value):
        self._conf.features.olink = value
    
    @property
    def olink_covariates(self):
        """Selection of UKB features to be used as covariates to the OLINK 
        dataset.
        """
        return self._conf.features.olink_covariates
    
    @olink_covariates.setter
    def olink_covariates(self, value):
        self._conf.features.olink_covariates = value
        

class ProcessingSettings(SettingsGroup):
    """Settings for (pre-)processing of data loaded from UK 
    Biobank.This is a configuration of various steps performed 
    to the data before it is fed to a predictive model.
    """
    
    @property
    def drop_features(self):
        """An (optional) list of features to drop. This setting is
        useful if you'd like to exclude particular features from 
        a feature group.
        """
        return self._conf.preproc.drop_cols
    
    @drop_features.setter
    def drop_features(self, value):
        self._conf.preproc.drop_cols = value
        
    @property
    def drop_patients_with_missing_frac(self):
        """Drop patients that have more missing feature values
        than this fraction. Values should be in range [0, 1]:
        - 0 means drop patients with at leas one missing value
        - 1 means never drop patients
        Letting patients through with missing feature values
        will result in missing value imputation (na_imputation
        setting).
        """
        return self._conf.preproc.drop_na_frac
    
    @drop_patients_with_missing_frac.setter
    def drop_patients_with_missing_frac(self, value):
        self._conf.preproc.drop_na_frac = value
        
    @property
    def correlation_thresh(self):
        """Minimum correlation level to apply when removing 
        correlated features in order to estimate feature effect 
        size.
        Values should be greater than zero (correlation value of
        0.2 is considered low) and smaller than or equal to one
        (in which case only identical features will be affected).
        """
        return self._conf.analysis.correlation_thresh
    
    @correlation_thresh.setter
    def correlation_thresh(self, value):
        self._conf.analysis.correlation_thresh = value
        
        
    @property
    def na_imputation(self):
        """Method used to fill out missing values of a feature.
        Possible values are:
          - mean
          - median 
          """
        return self._conf.preproc.na_imputation
    
    @na_imputation.setter
    def na_imputation(self, value):
        self._conf.preproc.na_imputation = value
        
    @property
    def feature_scaling(self):
        """Feature scaling is a data pre-processing stage
        that scales all values to possibly aid model fitting. 
        Possible values are:
        - standard : squash all values to the [0, 1] range
        - power : apply power transform (make value distribution 
          approximately normal)
        - None
        """
        return self._conf.preproc.scaling
    
    @feature_scaling.setter
    def feature_scaling(self, value):
        self._conf.preproc.scaling = value
        
        
class CohortSettings(SettingsGroup):
    """Patient cohort configuration settings. Cases can be
    defined either via ICD10 codes.
    """
    @property
    def training_controls(self):
        """Set of controls to use in training only. Avalable options are:
          - None - random sample of UK Biobank patients (excluding 
            those selected as cases)
          - list of subject IDs: specific custom control set
        """
        return self._conf.patients.training_controls
    
    @training_controls.setter
    def training_controls(self, value):
        self._conf.patients.training_controls = value
    
    @property
    def collapsing_controls(self):
        """Set of controls to use for the collapsing analysis. Use:
          - None - random sample of UK Biobank patients (excluding 
            those selected as cases)
          - list of subject IDs: specific custom control set
        """
        return self._conf.patients.collapsing_controls
    
    @collapsing_controls.setter
    def collapsing_controls(self, value):
        self._conf.patients.collapsing_controls = value
        
    @property
    def icd10_codes(self):
        """List of (full) ICD10 codes to use for selecting cases.
        """
        value = self._conf.patients.spec 
        if not isinstance(value, str):
            return value
    
    @icd10_codes.setter
    def icd10_codes(self, value):
        self._conf.patients.spec = value
        

class ReportSettings(SettingsGroup):
    """Collection of settings describing generation of Milton's 
    HTML report.
    """
    
    @property
    def title(self):
        """Title of HTML report produced by Milton.
        """
        return self._conf.output.title
    
    @title.setter
    def title(self, value):
        self._conf.output.title = value
        
    @property
    def location(self):
        """File system folder into which HTML report is saved.
        By default it is the current folder. If the folder does 
        not exist, it is created
        """
        return self._conf.output.location
    
    @location.setter
    def location(self, value):
        self._conf.output.location = value
        
    @property
    def model_performance(self):
        """Whether the report should include model performance charts.
        """
        return self._conf.output.model_performance
    
    @model_performance.setter
    def model_performance(self, value):
        self._conf.output.feature_importance = value
        
    @property
    def feature_importance(self):
        """Whether the report should include feature importance charts.
        """
        return self._conf.output.model_performance
    
    @feature_importance.setter
    def feature_importance(self, value):
        self._conf.output.feature_importance = value
        
    @property
    def effect_sizes(self):
        """Whether the report should include effect size charts.
        """
        return self._conf.output.effect_sizes
    
    @effect_sizes.setter
    def effect_sizes(self, value):
        self._conf.output.effect_sizes = value
        
    @property
    def embedding(self):
        """Whether the report should include 2D embedding of the data.
        """
        return self._conf.output.embedding
    
    @embedding.setter
    def embedding(self, value):
        self._conf.output.embedding = value
        
    @property
    def feature_embeddings(self):
        """Whether the report should include 2D feature embedding charts.
        """
        return self._conf.output.feature_embeddings
    
    @feature_embeddings.setter
    def feature_embeddings(self, value):
        self._conf.output.feature_embeddings = value
        
        
class AnalysisSettings(SettingsGroup):
    """Collection of settings describing options specific to 
    how Milton performs its analys.
    """
    
    @property
    def data_location(self):
        """Where to look for all data sets.
        """
        return self._conf.dataset.location
    
    @data_location.setter
    def data_location(self, value):
        self._conf.dataset.location = value
    
    @property
    def default_model(self):
        """Classifier used to estimate feature importances.
        
        Valid values are: 
        - 'xgb' (extreme gradient boosting)
        - 'random-forest'
        """
        return self._conf.analysis.default_model
    
    @default_model.setter
    def default_model(self, value):
        self._conf.analysis.default_model = value
        
    @property
    def gene_significance_thresh(self):
        """Stores results of collapsing analysis for genes with 
        enrichment p-value below this threshold.
        """
        return self._conf.analysis.gene_sign_thresh
    
    @gene_significance_thresh.setter
    def gene_significance_thresh(self, value):
        self._conf.analysis.gene_sign_thresh = value
        
    @property
    def disease_prevalence(self):
        """Optional prevalence of the disease for which Milton is run. When
        specified, Milton will use that value to adjust prediction threshold
        for new cases.
        """
        return self._conf.analysis.disease_prevalence
    
    @disease_prevalence.setter
    def disease_prevalence(self, value):
        self._conf.analysis.disease_prevalence = value
        
        
class Settings:
    """Milton's High-level configuration. It is grouped
    into a number of sections, which comprise individual 
    settings that can be modified. You can learn more
    about each setting by checking its documentation.
    """
    
    def __init__(self, conf=None):
        self._conf = conf or Configuration()
        
    @cached_property
    def cohort(self):
        return CohortSettings(self._conf, 'cohort')
        
    @cached_property
    def features(self):
        return FeatureSettings(self._conf, 'features')
        
    @cached_property
    def processing(self):
        return ProcessingSettings(self._conf, 'processing')
    
    @cached_property
    def report(self):
        return ReportSettings(self._conf, 'report')
    
    @cached_property
    def analysis(self):
        return AnalysisSettings(self._conf, 'analysis')
    
    def copy(self):
        return Settings(Configuration.copy(self._conf))
    
    def __call__(self):
        return self._conf
    
    def _properties(self):
        return sorted(name for name, val in type(self).__dict__.items()
                      if isinstance(val, cached_property))
    
    def __repr__(self):
        chunks = [
            'Milton Settings',
            '---------------',
        ]
        
        for settings in self._properties():
            chunks.append(getattr(self, settings).__repr__())
        
        return '\n'.join(chunks)
