"""Utilities for batch runs on slurm clusters.
"""
import re
import logging
import pickle
import subprocess
from io import StringIO
from getpass import getuser
import pandas as pd
from time import sleep, time
from pathlib import Path
from tqdm import tqdm
from functools import cached_property
from typing import Optional, Callable, Sequence


from .icd10 import ICD10Tree
from .configuration import Settings
from .classification import MiltonException
from .analysis import Evaluator, make_template_env


class SlurmArray:
    """Python version of the SLURM job array.
    """
    SQUEUE_POLL_DELAY = 3 * 60 # seconds
    
    def __init__(self, job_name: str, batch_size: int):
        self.job_name = job_name
        self.batch_size = batch_size
        self.n_running = 0
        self._last_squeue_poll = 0 
        
    def submit(self, script: str):
        """Submits a job for execution. When the number of currently running
        jobs (or awaiting start) reaches the max batch size the function blocks
        until resources are available.
        """
        while self.n_running >= self.batch_size:
            dt = time() - self._last_squeue_poll
            if dt > self.SQUEUE_POLL_DELAY:
                self.n_running = len(self.squeue(self.job_name))
                self._last_squeue_poll = time()
            else:
                sleep(self.SQUEUE_POLL_DELAY - dt)
        self.run_sbatch(script)
        self.n_running += 1
    
    @staticmethod
    def squeue(job_name):
        """Executes squeue for the current user and given job name presenting
        the results in a pandas DataFrame.
        """
        txt = subprocess.check_output(
            ['squeue', '--user', getuser(), '--name', job_name]).decode('utf8')
        outp = '\n'.join(line.strip() for line in txt.split('\n'))
        return pd.read_csv(StringIO(outp), sep=r'\s+')

    @staticmethod
    def run_sbatch(txt: str):
        """Submits an sbatch job with given job description file.
        """
        result = subprocess.run(
            'sbatch', input=bytes(txt, 'utf8'), capture_output=True)
        retcode = result.returncode
        if retcode != 0:
            output = result.stdout.decode('utf8')
            raise MiltonException(
                f'Non-zero return code ({retcode}) when running sbatch. '
                f'Output: {output}')
            
    
class JobResults:
    def __init__(self, location):
        self.location = Path(location)
        self.name = self.location.name

    @property
    def settings(self):
        """Loads the job's settings (input specs)
        """
        with (self.location / 'settings.pickle').open('rb') as f:
            return pickle.load(f)
        
    @property
    def status(self):
        path = self.location / '__STATUS__'
        if path.exists():
            with path.open('r') as f:
                return f.read()
        return 'None'
        
    def __repr__(self):
        return f'MiltonResults[name: {self.name}, status: {self.status}]'
            

class BatchRun:
    """Batch runner class for running Milton on multiple cohorts.
    """
    SLURM_POLL_PERIOD = 5
    
    def __init__(self, name, settings, cohorts, 
                 cohort_settings: Optional[Callable]=None,
                 output_path='.',
                 use_slurm=True,
                 job_minutes=45,
                 scp_partition='core',
                 slurm_batch_size=1000,
                 dry_run=False,
                 extra_sbatch_args=None):
        self.name = name
        self.settings = settings.copy()
        self._conf = settings()
        self.cohorts = cohorts
        self.cohort_settings = cohort_settings
        self.failed_cohorts = []
        self.output_path = (Path(output_path) / name).absolute() 
        self.use_slurm = use_slurm
        self.job_minutes = job_minutes
        self.scp_partition = scp_partition
        self._templates = make_template_env()
        self._validate()
        self.scripts = []
        self._slurm_array = SlurmArray(self.name, slurm_batch_size)
        self._dry_run = dry_run
        self.extra_sbatch_args = extra_sbatch_args or {}
        
    def _validate(self):
        if len(self.cohorts) == 0:
            raise ValueError('At least one cohort has to be provided.')
        if not all(isinstance(c, (pd.Series, ICD10Tree, str)) 
                   for c in self.cohorts):
            raise ValueError('All cohorts must be either pd.Series objects or '
                             'ICD10 sub-trees.')
            
    def __call__(self):
        self.failed_cohorts = []
        self.output_path.mkdir(exist_ok=not self._dry_run)
        
        if self.use_slurm:
            self._enqueue_slurm_jobs()
        else:
            self._run_with_dask()
            
    @staticmethod
    def _cohort_name(cohort, n):
        if isinstance(cohort, pd.Series):
            return cohort.name or f'Cohort-{n}'
        elif isinstance(cohort, list):
            # assuming a list of ICD10 strings
            return '_'.join(cohort)
        elif isinstance(cohort, ICD10Tree):
            return cohort.name
        else:
            raise MiltonException(f'Unsupported cohort type: {type(cohort)}')
            
    def _adapt_settings(self, main_settings, cohort, n):
        job_settings = main_settings.copy()
        name = self._cohort_name(cohort, n)
        job_settings.report.location = str(self.output_path / name)
        job_settings.report.title = name
        if isinstance(cohort, (list, ICD10Tree)): 
            job_settings.cohort.icd10_codes = cohort
        if self.cohort_settings is not None:
            self.cohort_settings(name, job_settings)
        return job_settings
    
    def _run_with_dask(self):
        for i, cohort in enumerate(tqdm(self.cohorts)):
            try:
                settings = self._adapt_settings(self.settings, cohort, i)
                the_cohort = cohort if isinstance(cohort, pd.Series) else None
                ev = Evaluator(settings)
                # start processing
                ev(cohort=the_cohort)  
                ev.save_report()
            except Exception as ex:
                self.failed_cohorts.append(cohort)
                logging.exception('Error during processing of '
                                  f'{self._cohort_name(cohort, i)}: {ex}')
        
    def _enqueue_slurm_jobs(self):
        # SLURM jobs will use local scheduler
        conf = self.settings.copy()()
        conf.cluster.kind = 'local'
        conf.cluster.n_workers = 1
        conf.cluster.cores = 4
        conf.cluster.memory = '24G'
        conf.cluster.port = None
        main_settings = Settings(conf)
        for i, cohort in enumerate(tqdm(self.cohorts, desc='Queuing jobs')):
            try:
                if isinstance(cohort, pd.Series): 
                    raise NotImplementedError('Cohorts as pd.Series not '
                                              'implemented for SLURM-based '
                                              'batch runs.')
                settings = self._adapt_settings(main_settings, cohort, i)
                # Settings are the SLURM job's input
                job_path = self.output_path / settings.report.title
                job_path.mkdir(exist_ok=True)  # important to SLURM
                script = self._setup_sbatch(settings, job_path)
                if not self._dry_run:
                    self._slurm_array.submit(script)
            except Exception as ex:
                self.failed_cohorts.append(cohort)
                logging.exception('Error during processing of '
                                  f'{self._cohort_name(cohort, i)}: {ex}')

    def _setup_sbatch(self, settings, job_path):
        settings_path = job_path / 'settings.pickle'
        with settings_path.open('wb') as f:
            pickle.dump(settings, f)
        template = self._templates.get_template('slurm_sbatch.jinja2')
        txt = template.render(name=self.name,
                              minutes=self.job_minutes,
                              scp_partition=self.scp_partition,
                              conf=settings(), 
                              result_path=job_path,
                              extra_params=self.extra_sbatch_args)
        self.scripts.append(txt)
        return txt
        
    @property
    def results(self):
        """Result tracker for this batch run, describing its current state.
        """
        return BatchRunResults(self.output_path)
    
    
class BatchRunResults:
    """Encapsulates batch run result set by providing functionality
    for keeping track of completed/failed/running job counts, iteration
    over individual job results and awaiting completion of the entire batch.
    """
    def __init__(self, location):
        self.location = Path(location)
        self.status = {}
        self.update_status()
        
    def update_status(self):
        """Reloads status information of this result set.
        """
        for job in self.location.iterdir():
            if self.status.get(job) is None and job.is_dir():
                self.status[job] = None
                try:
                    with (job / '__STATUS__').open('r') as f:
                        txt = f.readlines(1)[0]
                        self.status[job] = int(txt == 'SUCCESS')
                except Exception:
                    # in case the file doesn't exist 
                    pass

    @property
    def completed(self):
        """The number of completed jobs.
        """
        return sum((v == 1) for v in self.status.values())
    
    @property
    def failed(self):
        """The number of failed jobs so far.
        """
        return sum((v == 0) for v in self.status.values())
    
    @property
    def running(self):
        """The number of still running jobs.
        """
        return sum((v is None) for v in self.status.values())
    
    def iter_completed(self, subset=None):
        """Iterates over completed job results.
        
        Parameters
        ----------
        subset : (optional) iterable of as subset of job names to 
            iterate through.
        """
        jobs = self.status 
        if subset is not None:
            subset = set(subset)
            jobs = {k: v for k, v in jobs.items() if k.name in subset}
            
        return iter(FinishedJobResults(loc) 
                    for loc, status in jobs.items()
                    if status == 1)
    
    def iter_failed(self):
        """Iterates over failed job results.
        """
        return iter(FailedJobResults(loc) 
                    for loc, status in self.status.items()
                    if status == 0)
    
    def iter_unfinished(self):
        """Iterates over unfinished (killed by slurm) job results.
        """
        return iter(UnfinishedJobResults(loc) 
                    for loc, status in self.status.items()
                    if status is None)
        
    def __repr__(self):
        return '\n'.join([
            f'Batch {self.location.name}:',
            f'Completed tasks: {self.completed}',
            f'Failed tasks: {self.failed}',
            f'Running tasks: {self.running}'
        ])
    
    def await_completion(self):
        """Reentrant method that will block until all jobs have finished
        while showing a progress bar. Feel free to interrupt it when needed
        and to re-run later to show the progress bar again.
        """
        t0 = time()
        self.update_status()
        update_time = time() - t0
        some_time = max(1, 2 * update_time)
        
        n = len(self.status)
        current = self.completed + self.failed
        progress = tqdm(initial=current, total=n)
        
        try:
            while True:
                new = n - self.running
                if new > current:
                    progress.update(new - current)
                    current = new
                elif self.running == 0:
                    break
                sleep(some_time)
                self.update_status()
        except KeyboardInterrupt:
            # don't show stack trace in jupter
            pass
        finally:
            progress.close()
            return self
            
    def failure_reasons(self):
        """Creates a data frame with all failed ICD10 codes
        along with the reason text extracted from the logs.
        """
        rows = []
        for failure in self.iter_failed():
            reason = failure.reason.split('.')[0] if failure.reason else None
            rows.append((failure.name, reason))
            
        return pd.DataFrame(rows, columns=['name', 'reason'])\
            .set_index('name')['reason']
            
    def rerun(self, 
              jobs: Sequence[JobResults], 
              *,
              batch_size=200,
              job_minutes=45,
              scp_partition='core'):
        jobs = list(jobs)
        if not jobs:
            return
        # recover run name from the path of first job
        run_name = jobs[0].location.parent.name
        template = make_template_env().get_template('slurm_sbatch.jinja2')
        slurm_array = SlurmArray(run_name, batch_size)
        for job in tqdm(jobs):
            script = template.render(
                name=run_name,
                minutes=job_minutes,
                scp_partition=scp_partition,
                conf=job.settings(), 
                result_path=job.location,
                extra_params={})
            slurm_array.submit(script)


class FinishedJobResults(JobResults):
    """Wrapper for a completed successful job run. Has functionality
    for reading various result datasets.
    """
    
    def ftimp(self, models='General', replicas=False):
        """Loads feature importance estimated by this job for
        all or selected models (general, gender specific, linear, etc.)
        
        Parameters
        -----------
        models : model name, or name list of models to load feature 
            importance for. None means all models.
        """
        fpath = self.location / 'model_coeffs.csv'
        if isinstance(models, str):
            models = [models]
        if replicas: 
            cols = None  # all columns
        else:
            cols = ['Feature'] + models
        df = pd.read_csv(fpath, usecols=cols, index_col='Feature')
        replicas = df.columns[df.columns.str.startswith('replica')].to_list()
        return df[(models or []) + replicas]
    
    def has_qvsig(self):
        """Checks if there are collapsing analysis results available in this
        result set. Collapsing results may be missing in low quality models for
        which none of the predictions exceeded the minimum threshold.
        """
        return (self.location / 'qv_significance.parquet').exists()
    
    def qvsig(self, columns=None):
        """Loads QV significance results.
        
        Parameters
        ----------
        columns : load only selected columns (like pValue). None 
            loads all columns.
            
        Returns
        -------
        DataFrame with the results or None if data file does not exist
        """
        if columns is not None:
            columns = list({'cohort', 'QV model', 'gene'}.union(set(columns)))
            
        fpath = self.location / 'qv_significance.parquet'
        if fpath.exists():
            return pd.read_parquet(fpath, columns=columns)
    
    def metrics(self):
        """Loads the job's basic metrics.
        """
        df = pd.read_csv(self.location / 'metrics.csv').dropna()
        return df
    
    def scores(self, columns=None, replicas_only=False):
        """Load the job's prediction scores for the whole of UKB.
        """
        cols = None if replicas_only else columns
        df = pd.read_parquet(self.location / 'scores.parquet', columns=cols)
        if replicas_only:
            return df.loc[:, df.columns.str.startswith('replica')]
        else:
            return df
    
    def stuff(self):
        """Load the "stuff" file with various additional objects.
        """
        with (self.location / 'stuff.pickle').open('rb') as f:
            return pickle.load(f)
    
    
class FailedJobResults(JobResults):
    """Wrapper for failed Milton job results. The class alows
    for quick inspection of the job's logs, failure reason, 
    as well as it enables re-running the job in the context
    of the current session.
    """
    
    @cached_property
    def log(self):
        return (self.location / 'log.txt').read_text()
        
    @cached_property
    def reason(self):
        """Textual reason for failure: exception message of the 
        top-level exception that resulted in job failure.
        """
        for line in self.log.split('\n'):
            matches = re.findall('Failure reason: (.*)$', line)
            if not matches:
                matches = re.findall(
                    r'slurmstepd: error: .* \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
                    r' ([A-Za-z\s]+) \*', line)
            if matches:
                return matches[0]
        return 'Unknown'
                
        
    def rerun(self, save_report=False):
        """Reruns the job in the current dask cluster (the current 
        session).
        """
        self.job = Evaluator(self.settings)
        self.job.run()
        
        if save_report:
            self.job.save_report()
            status_file = self.location / '__STATUS__'
            with status_file.open('w') as f:
                f.write('SUCCESS')

        
class UnfinishedJobResults(FailedJobResults):
    """Wrapper for unfinished Milton job results. Unfinished jobs
    have most likely been killed by SLURM due to resource overuse
    (either time or memory). 
    """
    
    @cached_property
    def reason(self):
        for line in self.log.split('\n'):
            matches = re.findall('slurm.*error: (.*)$', line)
            if matches:
                return matches[0]
