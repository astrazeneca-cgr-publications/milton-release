import warnings
warnings.filterwarnings('ignore', 
                        category=FutureWarning,
                        module='^(dask_jobqueue|xgboost)')
import sys
import logging
from typing import Optional
from dask_jobqueue import SLURMCluster
from distributed import Client, LocalCluster

from .data_source import (
    UKB_DIR,
    UkbDataStore,
    UkbDataset, 
    UKB_DATA_LOCATION,
    DerivedBiomarkers)
from .data_desc import Col, UkbDataDict
from .data_info import UKB
from .patsel import UkbPatientSelector
from .globs import DD, DS, DASK, SEL, DST
from .configuration import Settings, Configuration, ModelType
from .icd10 import ICD10Tree


__all__ = [
    'Session',
    'UkbDataset', 
    'UkbPatientSelector', 
    'DerivedBiomarkers',
    'Col',
    'DD', 
    'DS', 
    'SEL',
    'DST',
    'ICD10',
    'Settings',
    'Configuration',
    'ModelType',
    'DASK_PORT',
]

# default port of the Dask scheduler
DASK_PORT = 8100

# ICD10 hierarchy
ICD10 = ICD10Tree.load()


class DaskResource:
    """Utility to create Dask clusters of various types via a simplified 
    interface.
    """
    def __init__(self, 
                 kind: str = f'tcp://localhost:{DASK_PORT}',
                 port: Optional[int] = DASK_PORT,
                 n_workers: int = 8,
                 memory: str = '4G',
                 cores: int = 2,
                 hours: int =2, 
                 partition='core', 
                 dashboard_port: Optional[int] = None,
                 **init_args):
        self.shared = None
        self.port = port
        self.dashboard_port = dashboard_port
        if kind == 'slurm':
            self.klass = SLURMCluster
            self.init_args = init_args | {
                'cores': cores,
                'memory': memory,
                'n_workers': n_workers,
                'job_cpu': 1,
                'job_mem': memory,
            }
            opts = self.init_args.get('scheduler_options', {})
            if self.dashboard_port:
                opts['dashboard_address'] = f':{self.dashboard_port}'
            if self.port: 
                opts['port'] = self.port
            if opts:    
                self.init_args['scheduler_options'] = opts
            self.init_args['job_extra'] = [
                f'-t {hours}:00:0',
                f'-p {partition}',
            ]
        elif kind == 'local':
            self.klass = LocalCluster
            self.init_args = {
                'n_workers': n_workers,
                'threads_per_worker': cores,
                'processes': True,  # multiprocessing
            }
            if self.port:
                self.init_args['scheduler_port'] = self.port
            if self.dashboard_port:
                self.init_args['dashboard_address'] = f':{self.dashboard_port}'
        elif kind.startswith('tcp://'):
            # connect to a running instance - no initialization needed
            self.shared = kind
        else:
            raise ValueError('Unsupported Dask cluster kind: %s' % kind)
        
    def start(self, wait=True): 
        if not self.shared:
            self.cluster = self.klass(**self.init_args)
            self.client = Client(self.cluster)
        else:
            self.client = Client(self.shared)
            self.cluster = None
        
        if wait:
            self._await()
            
        return self.cluster  
    
    def stop(self):
        self.client.close()
        if not self.shared:
            self.cluster.close()
        
    def scale(self, n_workers):
        if not self.shared:
            self.cluster.scale(n_workers)
            self._await()
        else:
            logging.warn('Scaling shared cluster not permitted. Ignoring.')
        
    def restart(self, wait=True):
        self.client.restart()
        if wait:
            self._await()
        
    def _await(self):
        if not self.shared:
            n = len(self.cluster.workers)
            self.client.wait_for_workers(n)
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, type, value, traceback):
        self.stop()
        return False
    
    def _repr_html_(self):
        return self.client._repr_html_()
    
    
class Session:
    """Session objects encapsulate a Dask cluster instance which runs all 
    Milton's computations and is a main entry point to Milton's functionality. 
    In a typical workflow you should create *only one* object of this class.
    """
    def __init__(self, 
                 kind: str = f'tcp://localhost:{DASK_PORT}',
                 port: Optional[int] = DASK_PORT,
                 n_workers: int = 8,
                 memory: str = '4G',
                 cores: int = 2,
                 data_location: str = UKB_DATA_LOCATION,
                 log_level=logging.WARNING,
                 interactive=True,
                 caching=True,
                 **dask_args):
        """Creates a new Milton session. Parameters define 
        the specifics of Dask cluster configuration.
        
        Parameters
        ----------
        kind : one of: 'slurm', 'local' or 'tcp://<address>:<port>,
          Type of cluster to start. The last is a URL to an already running
          scheduler.
        n_workers : int,
          Number of Dask workers to spin up.
        memory : string,
          Amount of memory per worker, typical value: '4G'.
        cores : int,
          Number of CPU cores to request for each worker.
        data_location : string,
          Path to Milton data files.
        interactive : boolean,
          Interactive session will set up the following global objects: 
          - DST: data store - collection of all UKB data sources,
          - DS: UKB data source, 
          - DD: UKB data dictionary, 
          - SEL: UKB patient selector.
        caching : boolean,
          Whether data access should be cached. A good rule of thumb is to use 
          caching only with SLURM Dask clusters.
        dask_args : any,
          extra keyword arguments to Dask cluster initializer.
        """
        logging.basicConfig(
            stream=sys.stdout,
            level=log_level,
            format='[%(asctime)s] %(message)s')
        
        self.caching = caching
        self.interactive = interactive
        self.data_location = data_location

        self._start_dask(kind=kind, 
                         n_workers=n_workers, 
                         memory=memory, 
                         cores=cores, 
                         port=port,
                         **dask_args)
        if interactive:
            self._create_globals()
            
    def _start_dask(self, **dask_args):
        logging.info('Connecting to Dask cluster.')
        self.cluster = DaskResource(**dask_args)
        self.cluster.start()
        DASK.set_global(self.cluster.client)
            
    def _create_globals(self):
        logging.info(f'Data caching is: {"ON" if self.caching else "OFF"}.')
        dd = UkbDataDict(self.data_location, UKB_DIR)
        dst = UkbDataStore(location=self.data_location, 
                           cached=self.caching)
        ds = UkbDataset(location=self.data_location,
                        dataset=dst[UKB].dataset, 
                        data_dict=dd)
        sel = UkbPatientSelector(data_store=dst,
                                 data_dict=dd)
        DST.set_global(dst)
        DS.set_global(ds)
        DD.set_global(dd)
        SEL.set_global(sel)
    
    def new_settings(self):
        """Returns a new settings object that is pre-initialized with
        session-specific information (data location, etc.)
        """
        settings = Settings()
        settings.analysis.data_location = self.data_location
        return settings
    
    def evaluate(self, settings, debug=False):
        """Evaluates a particular configuration specified via Milton's settings.
        
        Parameters
        ----------
        settings : Settings
          Global configuration to evaluate.
        """
        from milton.analysis import Evaluator
        ev = Evaluator(settings)
        try:
            ev.run()
        except Exception as ex:
            logging.error(ex)
            if debug or not self.interactive:
                raise
        return ev
        
    def eval_custom(self, 
                    features=None, 
                    cohort=None,
                    settings=None,
                    debug=False):
        """Custom interface to Milton's analysis.
        
        Returns
        -------
        Evaluation object with the results.
        """
        from milton.analysis import Evaluator
        settings = settings or Settings()
        ev = Evaluator(settings)
        try:
            ev.run(features=features, cohort=cohort)
        except Exception as ex:
            logging.error(ex)
            if debug or not self.interactive:
                raise
        return ev
                
    def run_batch(self, 
                  name,
                  settings, 
                  cohorts, 
                  use_slurm=True, 
                  debug=False, 
                  **run_args):
        """Queues up a batch run for execution. Batch runs are used to 
        evaluate a configuration against a (possibly large) set of patient
        cohorts.
        
        Parameters
        ----------
        name : str
          Name of the batch job. It should be descriptive of the configuration 
          used. Note that it will be made into file system folder name so make 
          sure it contains path-valid characters.
        settings : Settings
          Milton Settings object to use.
        cohorts : list
          List of cohorts, each being either a boolean pandas Series with 
          patient IDs in index and True values denoting cases and False denoting
          controls; or a list of ICD10 subtrees which will be used  to define 
          cohort's cases, while controls being defined in the settings.
        use_slurm : boolean
          When True, independend SLURM jobs will be queued up for processing of 
          each cohort. Default Dask cluster is used for all computations 
          otherwise.
            
        Returns
        -------
        
        BatchRun object with API for tracking the progress and processing of 
        the results.
        """
        from milton.batch import BatchRun
        batch = BatchRun(name, settings, cohorts, use_slurm, 
                         **run_args)
        try:
            batch()
        except Exception as ex: 
            logging.error(ex)
            if debug or not self.interactive:
                raise
        return batch
        
    def stop(self):
        """Tears down session-wide resources (dask cluster).
        """
        self.cluster.stop()
        
    def _repr_html_(self):
        return self.cluster._repr_html_()
