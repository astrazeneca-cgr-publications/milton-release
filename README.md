# MILTON: MachIne Learning PhenoType associatONs

**MILTON** is an automated framework for learning phenotype contributing factors from UK 
Biobank (UKB) structured data. The tool can also be used to define case-control 
cohorts based on continuous or binary traits contained in UKB or by prediction 
from the fitted machine learning models.

**Original Hypothesis & MILTON outcomes**: <br/>
Standard PheWAS analyses rely on ICD10 codes annotations for patient diagnosis. However, these may result to misdiagnosed patients or even undiagnosed ones (i.e. cryptic cases). We, thus, wanted to explore first whether we could identify those cryptic cases based on common biomarker signatures shared with already diagnosed patients. This would allow us to create predictive models to detect disease (prior or after onset) across **3,000 phenotypes**, using **blood** and **urine** based **biomarkers** as well as **plasma protein** expression levels. A follow-up outcome of this analysis was to construct augmented case cohorts and perform **enhanced PheWAS** analyses, allowing us to detect novel genetic signals.


# Installation

MILTON is not currently available via PyPi but it will install with pip in debug 
mode. For complete installation procedure that creates a conda env and installs
everything with pip, run:
```
scripts/init.sh
```
If you already have a conda env that you want to use, activate it and from
within the MILTON folder run:
```
pip install -r requirements.txt
pip install -e .
```

# Basic Usage

MILTON uses `dask` for parallel reading of partitioned datasets, hence it 
needs to start a session beforehand. 
```
from milton import *

sess = Session('local')  # use multiprocessing - most recommended
```
The session object offers a high-level interface for running MILTON pipelines 
based on pipeline configurations. The configurations are special objects that
have two-level hierarchy of settings and which define all aspects of a MILTON 
run. The `Settings` objects pretty-print their full contents in Jupyter 
notebooks so you can use that to learn the available options:
```
>> conf = Settings()
>> conf

Milton Settings
---------------
analysis:
  data_location : /home/.../dummy_ukb_data
  default_model : xgb
  disease_prevalence : None
  gene_significance_thresh : 0.05
cohort:
  collapsing_controls : None
  icd10_codes : None
  training_controls : None
features:
  biomarkers : False
  derived_biomarkers : None
  environmental : False
  lifestyle : False
  medical_history : False
  mental_health : False
  olink : False
  olink_covariates : False
  overall_health : False
  pulse_wave : False
  respiratory : False
  social : False
processing:
  correlation_thresh : 0.5
  drop_features : None
  drop_patients_with_missing_frac : 1.0
  feature_scaling : standard
  na_imputation : median
report:
  effect_sizes : True
  embedding : False
  feature_embeddings : None
  feature_importance : True
  location : .
  model_performance : True
  title : Milton Report
```
To view the full list of settings (lower-level) run:
```
conf()
```
By default, MILTON will run on the dummy UKB data included in the 
distribution. The following example sets up a dummy config to make it run on the
dummy data:
```
conf().analysis.min_cases = 10
conf().analysis.min_prediction_thresh = .5     # accept low quality predictions
conf().features.biomarkers = True              # default 67 biomarkers
conf().patients.spec = pd.Series(1, index=range(1, 51))  # dummy subject IDs for cases

# fit models, estimates feature importance
result = sess.evaluate(conf)  

# collapsing analysis and rest of pipeline
result.save_report('/path/to/folder')
```

# Milton Datasets

MILTON features a dedicated data access layer that requires data to be stored in
the `parquet` binary format. It uses `pandas` and `pyarrow` for data reading and 
processing. Check out the contents of the `dummy_ukb_data` folder to see how 
MILTON expects its datasets to be stored under a single root. The folder 
contains also companion files such as ancestry-specific sample ID files, ID 
lists for subjects that opted out from UKB and should not be included in 
analyses and collapsed variant matrices (UKB-derived data, see description 
below). Make sure you check the contents of the `data_info.py` file which 
defines the datasets expected to be found as well as the MILTON data root folder
under the `UKB_DATA_LOCATION` global constant.

UKB data comes by default as a collection of large CSV files which
are quite costly to work with directly, especially if you need to read them
multiple times. MILTON includes tools to convert the standard UKB data files 
into parquet. There are two routines, one converts the main UKB dataset which, 
due to the large number of columns, requires dedicated treatment and a more
generic function for conversion of remaining companion datasets.

## Conversion of Main UKB dataset

You will need a Dask cluster of several nodes, each with about 64 GB of RAM. 
The conversion if fully automatic and it sets up a small cluster with `slurm`:
```
from pathlib import Path
from milton.ukb_csv_to_parquet import *

convert_ukb_csv_to_parquet(
    Path('/path/to/ukb/release/main-ukb-file.csv'),
    n_chunks=8,
    output_path=Path(f'/path/to/output/folder/ukb.parquet'))
```

## Conversion of Remaining Datasets

The following example converts the `gp_clinical.txt` dataset to parquet and
splits it into a number of chunks for quicker reads. No dask cluster is used
but the full file is read to memory so make such you have enough of it:

```
from pathlib import Path
from pyarrow import csv
from milton.ukb_csv_to_parquet import *

ehr_csv_to_parquet(
    Path('/path/to/file/gp_clinical.txt'),
    Path('/path/to/output/gp_clinical.parquet'),
    read_options=csv.ReadOptions(
        block_size=2**30, 
        encoding='cp1252',
        use_threads=True),
    parse_options=csv.ParseOptions(delimiter='\t'),
    convert_options=csv.ConvertOptions(
        include_columns=[
            'eid', 'data_provider', 'event_dt', 'read_2', 'read_3',
        ],
        strings_can_be_null=True,
        auto_dict_encode=True,
        column_types={
           'value1': pa.string(),
           'value2': pa.string(),
        },
        timestamp_parsers=['%d/%m/%Y']),
    use_dictionary=['read_2', 'read_3'],
    write_statistics=False)
```

## Collapsed Variant Matrices

MILTON runs collapsing analysis on its extended cohorts and uses sparse matrices
(`csr_matrix` from scipy) to represent genotypes of all UKB subjects. Dummy 
examples are included in `qv_models` sub-folder of the dummy data folder and 
they comprise pickled triplets: subject IDs (matrix rows), gene names (matrix
columns), the csr_matrix object containing binary data that indicates presence/absence
of a variant in a gene for a subject.
The collapsed variant matrices were derived from UKB WES data by AZ Centre for
Genomics Research and more information can be found in the following two publications:
- https://doi.org/10.1038/s41586-021-03855-y
- https://doi.org/10.1038/s41576-019-0177-4



# Configuration Extras

Please note that when specifying your own case/control ids, MILTON doesn't know 
which ICD10 to use for time-lag calculation. Therefore, only time-agnostic model
is implemented in this case. Please perform the sub-setting yourself while 
deriving case and control ids.

## Cases and controls

**To specify multiple ICD10 codes**

```
from milton import *

desired_codes=['N18', 'C50', 'C61', 'F30']
all_codes_list=[]
for code in desired_codes:
    all_codes_list.append(list(ICD10.find_by_code(code)))
    
settings().patients.spec = list(itertools.chain(*all_codes_list))
```

**To specify your own list of cases and controls:**

- using case ids only

```settings().patients.spec = pd.Series(1, index=case_ids)```

- using case and control ids

```settings().patients.spec = pd.concat([pd.Series(1, index=case_ids), pd.Series(0, index=control_ids)])```

**To set minimum number of training cases to 0, default 100**

```settings().analysis.min_cases = 0```

**To specify control subset for training XGBoost**

```settings().patients.training_controls = <list of control ids>```

**To specify control subset for performing collapsing analysis**

```settings().patients.collapsing_controls = <list of control ids>```

**To remove certain subjects from analysis.**

```settings().patients.used_subjects = list(set(settings().patients.used_subjects).difference(<list of subject ids to exclude>)```
    
**To perform low power collapsing analysis (only on subjects with Olink or NMR metabolomics data, for example, and not the entire UKB cohort)**

```settings().analysis.collapsing_on_data_index = True```


## Features

**To run MILTON on a subset of proteins:**

```settings().features.olink = <list of olink protein names such as ['C2', 'TNF']>```

**To add extra features from UKB based on their field ids:**

```settings().features.ukb_custom = [21025, 21027] #UKB field IDs for additional 7 features```

## Custom feature imputation

```
settings().preproc.na_imputation_extra = {
            'Testosterone': GenderSpecNAStrategy(males='median', females='median'),
            Col.RHEUMATOID_FACTOR: ('constant', 0.0),
            Col.OESTRADIOL: GenderSpecNAStrategy(males=36.71, females=110.13),
            'Had menopause': CategoricalNAStrategy(),
            'Smoking status': CategoricalNAStrategy(),
            'Age when periods started (menarche)': ('constant', -5), 
            'Age at first live birth': ('constant', -5),
        }
```
# Feature Selection with Boruta

MILTON ships with version 0.4 of [boruta_py](https://github.com/scikit-learn-contrib/boruta_py) 
package since it cannot as of July 2024 be installed with `pip`. The implementation
is included verbatim. 
