"""UK Biobank Data description layer.
"""
from collections import defaultdict
from pyarrow.parquet import read_schema
from itertools import chain
from functools import wraps, cache
from pathlib import Path
import pandas as pd
import numpy as np
import time
import requests
from io import BytesIO
import re


INDEX_COL = 'eid'


class Col:
    """Collection of UKB column names that are often referenced.
    Also, it contains names of synthetic features created by UkbFeatureBuilder.
    """
    # general
    AGE = 'Age when attended assessment centre'
    AGE_RECRUIT = 'Age at recruitment'
    GENDER = 'Sex'

    # extra biomarkers
    CREATININE = 'Creatinine'
    UREA = 'Urea'
    URATE = 'Urate'
    MICROALBUMIN = 'Microalbumin in urine'
    CREATININE_IN_URINE = 'Creatinine (enzymatic) in urine'
    C_REACTIVE = 'C-reactive protein'
    ALT = 'Alanine aminotransferase'
    AST = 'Aspartate aminotransferase'
    PLATELETS = 'Platelet count'
    BMI = 'Body mass index (BMI)'
    WAIST = 'Waist circumference'
    HIP = 'Hip circumference'
    HEIGHT = 'Standing height'
    TRIGLYCERIDES = 'Triglycerides'
    GGT = 'Gamma glutamyltransferase'
    CHOLESTEROL = 'Cholesterol'
    HDL_CHOLESTEROL = 'HDL cholesterol'
    LDL_DIRECT = 'LDL direct'
    RHEUMATOID_FACTOR = 'Rheumatoid factor'
    OESTRADIOL = 'Oestradiol'
    VAT = 'Visceral adipose tissue volume (VAT)'
    ASAT = 'Abdominal subcutaneous adipose tissue volume (ASAT)'
    GLUCOSE = 'Glucose'
    GLYCATED_HAEMOGLOBIN = 'Glycated haemoglobin (HbA1c)'
    TESTOSTERONE = 'Testosterone'

    # overall health
    SYSTOLIC_BLOOD_PRESSURE = 'Systolic blood pressure'
    DIASTOLIC_BLOOD_PRESSURE = 'Diastolic blood pressure'
    PULSE_RATE = 'Pulse rate, automated reading'
    FASTING_TIME = 'Fasting Time'

    # lifestyle
    CANNABIS_FREQ = 'Maximum frequency of taking cannabis'

    # medical history
    END_STAGE_RENAL_DISEASE = 'Date of end stage renal disease report'
    OPERATIVE_PROCEDURES = 'Operative procedures - main OPCS4'

    # environmental
    AIR_POLLUTANTS = [
        'Nitrogen dioxide air pollution; 2010',
        'Nitrogen oxides air pollution; 2010',
        'Particulate matter air pollution (pm10); 2010',
        'Particulate matter air pollution (pm2.5); 2010'
    ]

    # social
    EMPLOYMENT = 'Current employment status'
    ETHNICITY = 'Ethnic background'

    # auxilliary
    DATE_MENTAL_HEALTH = 'Date of completing mental health questionnaire'
    YEAR_OF_BIRTH = 'Year of birth'
    AGE_LAST_TOOK_CANNABIS = 'Age when last took cannabis'

    # derived biomarkers (not in the original UKB dataset)
    EGFR_EPI = 'eGFR_EPI'
    EGFR_MDRD = 'eGFR_MDRD'
    UACR = 'UACR'
    ASTALT = 'ASTALT'
    FIB4 = 'FIB4'
    APRI = 'APRI'

    FLI = 'FLI'  # Fatty Liver Index
    HSI = 'HSI'  # Hepatic Steatosis Index
    BARD = 'BARD'  # BARD Score

    REMNANT_CHOLESTEROL = 'REMNANT_CHOLESTEROL'
    NON_HDL_CHOLESTEROL = 'NON_HDL_CHOLESTEROL'
    WAIST_HIP = 'WAIST_HIP'
    VAT_ASAT = 'VAT_ASAT'
    
    
class Fld:
    """UKB field ID constants for selected fileds. IDs are strings because
    it facilitates other operations (concatenation with instance IDs).
    """
    AGE_RECRUIT = '21022'
    AGE = '21003'
    ALT = '30620'
    ASAT = '22408'
    AST = '30650'
    BMI = '21001'
    BMI_v2 = '23104'  # another, less accurate way of measuring BMI
    CHOLESTEROL = '30690'
    CREATININE = '30700'
    CREATININE_IN_URINE = '30510'
    ETHNICITY = '21000'
    GENDER = '31'
    GGT = '30730'
    HDL_CHOLESTEROL = '30760'
    HIP = '49'
    HEIGHT = '50'
    LDL_DIRECT = '30780'
    MICROALBUMIN = '30500'
    PLATELETS = '30080'
    TRIGLYCERIDES = '30870'
    VAT = '22407'
    WAIST = '48'
    EMPLOYMENT = '6142'
    CANNABIS_FREQ = '20454'
    AGE_LAST_TOOK_CANNABIS = '20455'
    DATE_MENTAL_HEALTH = '20400'
    YEAR_OF_BIRTH = '34'
    MONTH_OF_BIRTH = '52'
    OPERATIVE_PROCEDURES = '41200'
    END_STAGE_RENAL_DISEASE = '42026'
    
    
@cache
def load_parquet_schema(path):
    return read_schema(str(path))


class UkbCatalogue:
    """Collection of UKB urls that point to meta information about the main
    UKB dataset.
    """
    
    SCHEMA = 'https://biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=1'
    CATEGORIES = 'https://biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=3'
    INT_ENCODINGS = 'https://biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=5'
    HIER_INT_ENCODINGS = 'https://biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=11'
    
    @classmethod
    @property
    def schema(cls):
        return cls.fetch_tab_separated(cls.SCHEMA, 'field_id')
    
    @classmethod
    @property
    def categories(cls):
        return cls.fetch_tab_separated(cls.CATEGORIES, 'category_id')
    
    @classmethod
    @property
    def int_encodings(cls):
        return cls.fetch_tab_separated(cls.INT_ENCODINGS, encoding='1252')
    
    @classmethod
    @property
    def hier_int_encodings(cls):
        return cls.fetch_tab_separated(cls.HIER_INT_ENCODINGS, encoding='1252')
    
    @staticmethod
    def fetch_tab_separated(url, index=None, encoding=None):
        """Fetches a TAB-separated file and makes it into a data frame.
        """
        resp = requests.get(url)
        err_msg = f'Unexpected status {resp.status_code} when fetching from {url}.'
        assert resp.status_code == 200, err_msg
        return pd.read_csv(
            BytesIO(resp.content),
            sep='\t',
            encoding=encoding,
            index_col=index).sort_index()


class UkbDataDict:
    """ Data Dictionary object that implements functionality for searching
    and querying the mapping between numerical column encodings (eg 10010-1.0)
    and their textual descriptions.
    
    It relies on meta information obtained from the UKB site. See the UkbUrl 
    class for details on which of the UKB catalogue The following is
    a list of URLs which provide the data, which has to be (as of 2020) downloaded
    manually, as tab-separated files. The files should all be put under the 
    Milton's UKB data location with their original names.
    
      - data field categories: https://biobank.ndph.ox.ac.uk/showcase/schema.cgi?id=3
      - data schema: https://biobank.ndph.ox.ac.uk/showcase/schema.cgi?id=1
      - simple int encodings: https://biobank.ndph.ox.ac.uk/showcase/schema.cgi?id=5
      - hierarchical int encodings: https://biobank.ndph.ox.ac.uk/showcase/schema.cgi?id=11
    """
    ASSAY_CATEGORIES = [
        'Urine assays',
        'Blood count',
        'Blood biochemistry'
    ]

    INFECTIOUS_DISEASES = 'Infectious Diseases'

    # these are fused into one to increase number of available data points
    BLOOD_PRESSURES = [
        'Systolic blood pressure, automated reading',
        'Diastolic blood pressure, automated reading',
    ]

    # extra features needed for feature building
    AUXILLIARY = [
        Col.DATE_MENTAL_HEALTH,
        Col.YEAR_OF_BIRTH,
        Col.AGE_LAST_TOOK_CANNABIS
    ]

    def __init__(self, location, ukb_dir):
        self.location = Path(location)
        if not (self.location / 'category.txt').exists():
            self.update_catalogues()
        self.df = self._load_data_field_desc(self.location, ukb_dir)
        self.categories = self._load_cat_encodings(self.location)
        self.field_names = self.df[['field_id', 'title']]\
            .set_index('field_id')['title']\
            .to_dict()
        self._build_feature_sets()
        self._build_feature_encodings()
        self._build_first_occ_mapping()
        
    def __getitem__(self, column_or_field):
        field = str(column_or_field)
        if '-' in field:
            field = field.split('-')[0]
        return self.field_names[int(field)]
    
    def maybe_rename(self, column_or_field):
        if re.match(r'[\d\.\-]', column_or_field):
            # must be field ID - rename it
            return self[column_or_field]  
        else:
            # must be a column name or something else
            return column_or_field
        
    def update_catalogues(self):
        """Fetches the most recent metadata files for the UKB dataset needed
        to construct this data dictionary.
        """
        UkbCatalogue.categories.to_csv(self.location / 'category.txt')
        UkbCatalogue.schema.to_csv(self.location / 'field.txt')
        UkbCatalogue.int_encodings.to_csv(self.location / 'esimpint.txt')
        UkbCatalogue.hier_int_encodings.to_csv(self.location / 'ehierint.txt')
    
    @staticmethod
    def _load_data_field_desc(location, ukb_dir):
        # data field categories
        categories = pd.read_csv(location / 'category.txt', 
                                 index_col='category_id',
                                 usecols=['category_id', 'title'])
        # field properties
        df = pd.read_csv(location / 'field.txt')\
            .join(categories['title'].rename('category'),
                  on='main_category', 
                  how='left')
        # field_id in columns is needed for searching, in index for convenience
        df.index = pd.Index(df['field_id'])
    
        # Add info about field instances (versions) - use the actual data schema
        pq_files = [p for p in (location / ukb_dir).iterdir()
                    if p.name.endswith('.parquet')]
        schema = load_parquet_schema(pq_files[0])
        columns = [f.name for f in schema if f.name != INDEX_COL]
        colmap = defaultdict(list)
        for c in columns:
            field_id, version = c.replace('--', '-').split('-')
            colmap[int(field_id)].append(version)
        df['instances'] = df['field_id'].map(colmap)
        return df
    
    def __getitem__(self, field_ids):
        if not isinstance(field_ids, list):
            raise ValueError('Expecting a list of field IDs (strings or ints)')
        return self.df.loc[[int(f) for f in field_ids]]
        
    @staticmethod
    def _load_cat_encodings(location):
        # Simple Integer Encodings
        # Found at: https://biobank.ndph.ox.ac.uk/showcase/schema.cgi?id=5
        simpint = pd.read_csv(location / 'esimpint.txt')
        
        # Hierarchical Integer Encodings
        # Found at: https://biobank.ndph.ox.ac.uk/showcase/schema.cgi?id=12
        hierint = pd.read_csv(location / 'ehierint.txt')
        
        encodings = (pd.concat([simpint, hierint])
                     # filter out negative values as encodings for NAs
                     .pipe(lambda df: df[df['value'] >= 0])  
                     .assign(value=lambda df: df['value'].astype('str')))
        
        # convert the encoding to a nested a map: 
        # encoding_id -> {code -> txt_value}
        # also, negative codes are removed since they represent 
        # different reasons for absent values so, by removing them here,
        # feature builder will put NAs in their place.
        return encodings.groupby('encoding_id')\
            .apply(lambda df: df.set_index('value')['meaning'].to_dict())\
            .to_dict()

    def _build_feature_sets(self):
        # blood count biomarkers come with two types of measurements:
        # the counts and percentages, except of Haematocrit which is
        # provided only as percentage
        self.assay_biomarkers = self \
            .find(category=self.ASSAY_CATEGORIES) \
            .dropna(subset=['units']) \
            .pipe(lambda df: df[(df['units'] != 'percent') |
                                (df['title'] == 'Haematocrit percentage')])

        self.other_biomarkers = pd.concat([
            self.find(title=Col.WAIST),
            # there are 2 BMI measurements (lergely correlated)
            self.find(title='bmi', notes='weight'),
            self.find(field_id=[int(Fld.HIP), int(Fld.HEIGHT)])
        ])

        self.biomarkers = pd.concat([
            self.assay_biomarkers,
            self.other_biomarkers
        ])
        
        self.respiratory = self.find(title=[
            'Forced expiratory volume in 1-second (FEV1), Best measure',
            'Forced vital capacity (FVC), Best measure',
            'FEV1/ FVC ratio Z-score'
        ])

        self.general_features = self.find(title=[
            Col.GENDER,
            Col.AGE
        ])

        self.lifestyle = self.find(title=[
            'Alcohol intake frequency.',
            'Ever addicted to any substance or behaviour',
            'Pack years of smoking',
            'Sleep duration',
            'Sleeplessness / insomnia',
            'Snoring',
            Col.CANNABIS_FREQ
        ])

        self.medical_history = pd.concat([
            self.find(category=self.INFECTIOUS_DISEASES, 
                      title='seropositivity'),
            self.find(title=[
                Col.END_STAGE_RENAL_DISEASE,
                Col.OPERATIVE_PROCEDURES,
            ])
        ])

        self.pulse_wave = pd.concat([
            self.find(title='pulse wave') \
                .pipe(lambda df: df[~df['title']
                                    .str.match('.*(curve|velocity).*')]),
            self.find(title=[
                'Cardiac index during PWA'
            ])
        ])

        self.overall_health = pd.concat([
            self.find(title=self.BLOOD_PRESSURES + [
                Col.PULSE_RATE,
                Col.FASTING_TIME,
            ]),
        ])

        self.mental_health = pd.concat([
            self.find(title=[
                'Worrier / anxious feelings',
                'Ever had prolonged feelings of sadness or depression',
            ]),
            self.find(category='mental health',
                      title='mental distress|seen.*anxiety',
                      regex=True),
        ])

        self.environmental = self.find(
            title=Col.AIR_POLLUTANTS + [
                'Average 24-hour sound level of noise pollution'
            ])

        self.social = pd.concat([
            self.find(title='^qualifications$', regex=True),
            self.find(title=[
                # those 2 features will be fused into one
                'Current employment status - corrected',
                'Current employment status',
                Col.ETHNICITY,
            ])
        ])
        
        self.olink_covariates = pd.concat([
            self.find(title=[
                'Illnesses of father',
                'Illnesses of mother',
                'Alcohol intake frequency.',
                'Smoking status', 
                'Blood-type haplotype']),
            self.find(field_id=int(Fld.BMI))
        ])

        self.biomarker_names = np.unique(self.biomarkers['title'])

        # all features extracted from UKB
        self.features = pd.concat([
            self.biomarkers,
            self.general_features,
            self.lifestyle,
            self.medical_history,
            self.pulse_wave,
            self.overall_health,
            self.mental_health,
            self.environmental,
            self.social,
            self.olink_covariates,
        ]).drop_duplicates(['field_id'])
        self.features.set_index('title', inplace=True)
        self.features.sort_index(inplace=True)

    def _build_feature_encodings(self):
        self.feature_encodings = {}
        encoded_features = self.df[self.df['encoding_id'] != 0]\
            .set_index('field_id')['encoding_id']
            
        assert encoded_features.index.is_unique, 'Field ID index is not unique.'
            
        for name, code in encoded_features.items():
            encoding = self.categories.get(code)
            if encoding is not None:
                self.feature_encodings[name] = encoding
                
    def _build_first_occ_mapping(self):
        """A mapping from 3-char ICD10 codes to the corresponding first 
        occurence field ID.
        """
        self.first_occurrence = self.find(title='first reported')['title']\
            .str.findall(r'^Date ([A-Z]\d\d) first reported')\
            .apply(lambda v: v[0])\
            .pipe(lambda s: pd.Series(s.index.astype('str'), index=s.values))\
            .to_dict()

    def predefined(self,
                   biomarkers=False,
                   respiratory=False,
                   lifestyle=False,
                   med_hist=False,
                   overall_health=False,
                   pulse_wave=False,
                   mental_health=False,
                   environmental=False,
                   social=False,
                   olink_covariates=False):
        """Convenience method for building a schema out of predefined
        feature groups.
        """
        parts = [self.general_features]
        needs_aux = False

        if biomarkers:
            parts.append(self.biomarkers)
            
        if respiratory:
            parts.append(self.respiratory)

        if lifestyle:
            parts.append(self.lifestyle)
            needs_aux = True

        if med_hist:
            parts.append(self.medical_history)

        if overall_health:
            parts.append(self.overall_health)

        if pulse_wave:
            parts.append(self.pulse_wave)

        if mental_health:
            parts.append(self.mental_health)

        if environmental:
            parts.append(self.environmental)

        if social:
            parts.append(self.social)
            
        if olink_covariates:
            parts.append(self.olink_covariates)

        if needs_aux:
            aux = self.find(title=self.AUXILLIARY) \
                .pipe(lambda df: df[~df['category'].str
                                    .startswith('Employment')])
            parts.append(aux)

        return self.to_schema(pd.concat(parts).drop_duplicates(['field_id']))

    def to_schema(self, filtered_df, instances=None, names=True):
        """Converts a dataframe of column descriptions (a subset
        of rows of UkbDataDict's data dictionary df) to a pandas
        Series with column names in index and versioned column
        IDs in values.

        Parameters
        ----------
        filtered_df : a subset of rows in the data dictionary.
        instances : None or a list of particular instances
            to request for *each* feature. Use None for all available
            instances (default).
        names : use field names instead of field IDs.
        """
        cols = []
        ix = []
        field_name = 'title' if names else 'field_id'

        for _, row in filtered_df.iterrows():
            if row['instances']:
                for v in row['instances']:
                    instance = int(v.split('.')[0])
                    if instances is None or instance in instances:
                        col_name = '-'.join([str(row['field_id']), v])
                        cols.append(col_name)
                        ix.append(str(row[field_name]))
            elif instances is None:
                # there are several non-instanced fields
                cols.append(row['field_id'])
                ix.append(str(row[field_name]))
            else:
                raise ValueError(
                    'Cannot select instances from unversioned field: '
                    f'{row[field_name]}.')
                
        return pd.Series(cols, index=ix)

    def find(self, regex=False, **field_values):
        def convert(v):
            return v.lower() if isinstance(v, str) else v

        condition = np.full(len(self.df), True, 'bool')

        for field, value in field_values.items():
            col = self.df[field]
            if col.dtype == 'object':
                col = col.str.lower()

            if isinstance(value, str):
                condition &= col.str.contains(value.lower(), regex=regex)
            elif isinstance(value, list):
                val_lst = [convert(v) for v in value]
                condition &= col.isin(val_lst)
            else:
                condition &= col == value

        return self.df[condition] 
    

class BiomarkerFunc:
    """Descriptor for a derived biomarker function. Its main obligation
    is to store the biomarker's dependencies in the provided store
    (a dictionary).
    """
    def __init__(self, func, store, dependencies):
        def wrapper(self_of_func):
            @wraps(func)
            def wrapped_func(*args, **kwargs):
                return func(self_of_func, *args, **kwargs)

            return wrapped_func

        self.func = wrapper
        self.store = store
        self.dependencies = sorted(set(dependencies))

    def __set_name__(self, owner, name):
        self.store[name] = self.dependencies

    def __get__(self, obj, obj_type=None):
        return self.func(obj)


def biomarker(store, *dependencies):
    """Decorator that enables annotation of biomarker-calculating methods
    with dependency information.
    """
    def wrapper(func):
        return BiomarkerFunc(func, store, dependencies)

    return wrapper


class DerivedBiomarkers:
    # name of set of patients with type-2 diabetes
    DEPENDENCIES = {}

    def __init__(self, location, biomarkers):
        """New DerivedBiomarkers instance.

        Parameters
        ----------
        data_dict : an instance of UkbDataDictionary
        biomarkers : list of biomarker names to calculate or True to indicate
            computation of all.
        """
        if biomarkers is True:
            self._select_all_biomarkers()
        else:
            self._select(biomarkers)
        self.location = Path(location)

    def _select_all_biomarkers(self):
        self.names = sorted(n.lower() for n in self.DEPENDENCIES)
        self.dependencies = sorted(set(chain(*self.DEPENDENCIES.values())))

    def _select(self, names):
        """Aggregates the data dependencies of selected biomarkers.
        """
        self.names = []
        deps = set()
        for name in names:
            if hasattr(self, name.lower()):
                self.names.append(name)
                deps.update(self.DEPENDENCIES[name.lower()])
            else:
                raise ValueError('Unrecognized derived biomarker: %s' % name)
        self.dependencies = sorted(deps)

    @classmethod
    def contains(cls, name):
        """Static list of all defined derived biomarkers.
        """
        return name.lower() in cls.DEPENDENCIES
    
    @classmethod
    def deps_of(cls, name):
        """Source fields required by the biomarker with given name.
        """
        return cls.DEPENDENCIES[name.lower()]
            
    def calculate(self, df, names=None):
        """Calculates all requested biomarkers from data in df. The data frame
        must contain all columns listed in the dependecies of each biomarker.

        Parameters
        ----------
        df : the data frame (must include all required columns)
        names : a list of biomarker names to calculate. Note, df is expected to 
          contain all required fields. When None, the default set from the 
          constructor is used.

        Returns
        -------
        New data frame with new columns added.
        """
        results = {}
        for name in names or self.names:
            biom = getattr(self, name.lower())
            results[name] = biom(df).rename(name, copy=False)

        out_df = pd.concat(results, axis=1)
        return out_df
    
    @biomarker(DEPENDENCIES,
               Fld.CREATININE, Fld.AGE_RECRUIT, Fld.ETHNICITY, Fld.GENDER)
    def egfr_mdrd(self, df):
        Scr = 0.0113 * df[Fld.CREATININE]  # unit: mg/dL
        age = df[Fld.AGE_RECRUIT].pow(-0.203)

        black = df[Fld.ETHNICITY] \
            .map({'Black or Black British': 1.212}) \
            .astype('float') \
            .fillna(1)

        gender = df[Fld.GENDER] \
            .map({'Female': 0.742}) \
            .astype('float') \
            .fillna(1)

        return 175 * Scr.pow(-1.154) * age * black * gender

    @biomarker(DEPENDENCIES,
               Fld.CREATININE, Fld.GENDER, Fld.ETHNICITY, Fld.AGE_RECRUIT)
    def egfr_epi(self, df):
        Scr = 0.0113 * df[Fld.CREATININE]  # unit: mg/dL
        kappa = df[Fld.GENDER].map({'Female': .7, 'Male': .9}).astype('float')
        alpha = df[Fld.GENDER].map({'Female': -0.329, 'Male': -0.411}).astype(
            'float')

        female = df[Fld.GENDER].map({'Female': 1.018})
        black = df[Fld.ETHNICITY].map({'Black or Black British': 1.159})

        return (141 * np.minimum(Scr / kappa, 1).pow(alpha)
                * np.maximum(Scr / kappa, 1).pow(-1.209)
                * np.power(0.993, df[Fld.AGE_RECRUIT])
                * female.astype('float').fillna(1)
                * black.astype('float').fillna(1))

    @biomarker(DEPENDENCIES, Fld.CREATININE_IN_URINE, Fld.MICROALBUMIN)
    def uacr(self, df):
        Scr = 0.0113 * df[Fld.CREATININE_IN_URINE]  # unit: mg/dL
        m_albumin = 0.1 * df[Fld.MICROALBUMIN]
        creatinine = Scr / 1000  # unit: g/dL
        return m_albumin / creatinine

    @biomarker(DEPENDENCIES, Fld.AST, Fld.ALT)
    def astalt(self, df):
        return df[Fld.AST] / df[Fld.ALT]

    @biomarker(DEPENDENCIES, Fld.AGE_RECRUIT, Fld.AST, Fld.ALT, Fld.PLATELETS)
    def fib4(self, df):
        a = df[Fld.AGE_RECRUIT] * df[Fld.AST]
        b = df[Fld.PLATELETS] * np.sqrt(df[Fld.ALT])
        return a / b

    @biomarker(DEPENDENCIES, Fld.AST, Fld.PLATELETS)
    def apri(self, df):
        return 100 * (df[Fld.AST] / 40) / df[Fld.PLATELETS]

    @biomarker(DEPENDENCIES, Fld.TRIGLYCERIDES, Fld.BMI, Fld.BMI_v2, Fld.GGT, Fld.WAIST)
    def fli(self, df):
        """Fatty Liver Index
        https://www.mdcalc.com/fatty-liver-index
        """
        y = (0.953 * np.log(18.02 * df[Fld.TRIGLYCERIDES])
             + 0.139 * df[Fld.BMI]
             + 0.718 * np.log(df[Fld.GGT])
             + 0.053 * df[Fld.WAIST]) - 15.745
        exp_y = np.exp(y)
        return 100 * exp_y / (1 + exp_y)

    @biomarker(DEPENDENCIES,
               Fld.CHOLESTEROL, Fld.LDL_DIRECT, Fld.HDL_CHOLESTEROL)
    def remnant_cholesterol(self, df):
        return df[Fld.CHOLESTEROL] - (df[Fld.LDL_DIRECT]
                                      + df[Fld.HDL_CHOLESTEROL])

    @biomarker(DEPENDENCIES, Fld.CHOLESTEROL, Fld.HDL_CHOLESTEROL)
    def non_hdl_cholesterol(self, df):
        return df[Fld.CHOLESTEROL] - df[Fld.HDL_CHOLESTEROL]

    @biomarker(DEPENDENCIES, Fld.VAT, Fld.ASAT)
    def vat_asat(self, df):
        return df[Fld.VAT] / df[Fld.ASAT]

    @biomarker(DEPENDENCIES, Fld.WAIST, Fld.HIP)
    def waist_hip(self, df):
        return df[Fld.WAIST] / df[Fld.HIP]
    
    @biomarker(DEPENDENCIES, Fld.YEAR_OF_BIRTH, Fld.MONTH_OF_BIRTH)
    def date_of_birth(self, df):
        no_data = df[Fld.MONTH_OF_BIRTH].isna() | df[Fld.YEAR_OF_BIRTH].isna()
        dates = pd.DataFrame({
            'day': 15,
            'month': df[Fld.MONTH_OF_BIRTH].cat
                .rename_categories(lambda txt: time.strptime(txt, '%B').tm_mon)
                .fillna(1)
                .astype('int8'),
            'year': df[Fld.YEAR_OF_BIRTH]
        })
        return pd.to_datetime(dates).mask(no_data, pd.NA)
