"""Scripting interface to MILTON. Used by batch jobs on slurm clusters.
"""
import argparse
import pickle
from time import time
import logging
from pathlib import Path

from milton import Session


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MILTON - MachIne Learning PhenoType associatON')
    
    parser.add_argument('settings', metavar='s', type=str, nargs=1,
                       help='Path to the configuration/settings object')
    try:
        args = parser.parse_args()
        path = args.settings[0]
        with open(path, 'rb') as f:
            job = pickle.load(f)
    except Exception as ex:
        logging.exception(f'Cannot find settings under {path}.')
        raise
        
    try:
        success = True
        location = Path(job.report.location)
        location.mkdir(exist_ok=True)
        
        t0 = time()
        cluster = job().cluster
        
        sess = Session(kind=cluster.kind,
                       n_workers=cluster.n_workers,
                       cores=cluster.cores,
                       memory=cluster.memory,
                       port=cluster.port,
                       interactive=False,
                       caching=False,
                       log_level=logging.INFO)

        logging.info('Evaluating configuration.')
        ev = sess.evaluate(job)

        logging.info('Saving results.')
        ev.save_report()
        
    except Exception as ex:
        # this line is searched for by the result collector
        logging.error(f'Failure reason: {ex}')  
        logging.exception(ex)
        success = False
    finally:
        dt = time() - t0
        logging.info(f'Finished in {int(dt)} seconds.')
        
        # write the STATUS file to indicate success
        status_file = Path(job.report.location) / '__STATUS__'
        with status_file.open('w') as f:
            f.write('SUCCESS' if success else 'FAILURE')
        sess.stop()
