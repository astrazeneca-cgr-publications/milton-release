"""Meta information about datasets comprising Milton's UKB datasets.
"""
from enum import Enum
from pathlib import Path

UKB = 'ukb'
BIOMARKERS = 'biomarkers'
HESIN_DIAG = 'hesin_diag'
HESIN_OPER = 'hesin_oper'
GP_CLINICAL = 'gp_clinical'
GP_SCRIPTS = 'gp_scripts'
DEATH_CAUSE = 'death_cause'
OLINK = 'olink'
CONSENSUS = 'consensus'

DATASETS = [
    UKB,
    OLINK,
    HESIN_DIAG,
    HESIN_OPER,
    GP_CLINICAL,
    GP_SCRIPTS,
    DEATH_CAUSE,
]

class Orient(Enum):
    """Dataset orientation: 
    - WIDE means unique index, multiple field values per index entry stored as 
      columns
    - TALL means non-unique index, multiple field values per index entry stored 
      in multiple rows (not necessarily sequentially or in order)
    Both tall and wide datasets may contain multiple fields. 
    """
    WIDE = 0
    TALL = 1

# default folder name with the UKB data
UKB_DIR = 'ukb.parquet'
    
# default name of the file with ID list of subjects who opted-out from the UKB
UKB_OPT_OUT_SUBJECTS = 'ukb-opt-outs.csv'


def _get_dummy_data_location():
    pkg = Path(__file__).absolute().parent.parent
    return str(pkg / 'dummy_ukb_data')

# default location of the data: replace that with a path to real data
UKB_DATA_LOCATION = _get_dummy_data_location()
