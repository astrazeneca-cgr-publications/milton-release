"""Automatic conversion of large UKB CSV dumps to parquet.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
import re
import subprocess
from dask import delayed
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import csv
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from milton.data_desc import UkbCatalogue
from tqdm import tqdm


def load_csv_partition(path,
                       dtype_map,
                       col_names,
                       non_date_cols,
                       timestamp_cols,
                       nrows=None):
    is_first = re.match('.*-0+$', path)
    return pd.read_csv(
        path,
        usecols=list(dtype_map),
        names=col_names,
        skiprows=1 if is_first else None,
        header=None,
        dtype=non_date_cols,
        parse_dates=timestamp_cols,
        infer_datetime_format=True,
        low_memory=True,
        encoding='cp1252',
        nrows=nrows)


def csv_to_parquet(path,
                   out_dir,
                   dtype_map,
                   col_names,
                   non_date_cols,
                   timestamp_cols,
                   nrows=None):
    file_id = re.findall(r'csv-split-(\d+)', path.name)[0]
    out_fname = 'part-' + file_id + '.parquet'
    out_path = Path(out_dir) / out_fname

    df = load_csv_partition(str(path), dtype_map, col_names, 
                            non_date_cols, timestamp_cols, nrows)
    table = pa.Table.from_pandas(df)
    pq.write_table(table,
                   out_path,
                   use_dictionary=False,
                   write_statistics=False)
    return out_path


def spinup_dask_cluster(n_workers):
    cluster = SLURMCluster(
        n_workers=n_workers,
        name='ukb-to-parquet',
        memory='128G',
        cores=1,
        job_extra=['-t 2:00:0'])

    client = Client(cluster)
    return client, cluster


def count_lines(file_path):
    out = subprocess.run(['wc', '-l', str(file_path)], 
                         check=True, 
                         capture_output=True, 
                         text=True)
    return int(out.stdout.split(' ')[0])


def split_file(file_path, n_chunks, out_dir):
    n_lines = count_lines(file_path)
    lines_per_chunk = int(np.ceil(n_lines / n_chunks))
    name_suffix_len = int(np.ceil(np.log10(n_chunks)))
    out_path = out_dir / file_path.name
    
    cmd = f'split -l {lines_per_chunk} {file_path} {out_path}-split- -da {name_suffix_len}'
    subprocess.run(cmd.split(' '))


def remove_csv_parts(orig_csv_file):
    orig_path = Path(orig_csv_file)
    for path in orig_path.parent.iterdir():
        if path.name.startswith(orig_path.name):
            if re.match(r'-split-\d+', path.name[len(orig_path.name):]):
                print('Removing:', path)
                path.unlink()
                
                
def convert_ukb_csv_to_parquet(csv_path,
                               n_chunks,
                               output_path,
                               cleanup=True):
    """
    The function implements conversion of a large UKB CSV dump file to parquet.
    In order to correctly assign data types to parquet columns, a schema file
    is obtained from UKB:

    * https://biobank.ndph.ox.ac.uk/showcase/schema.cgi?id=1

    The file defines data types for all UKB fields. The data types are encoded
    as integers. The following data types are present in UKB
    (source: https://biobank.ndph.ox.ac.uk/showcase/list.cgi):

    * integer - 11
    * Categorical (single) - 21
    * Categorical (multiple) - 22
    * Continuous - 31
    * Text - 41
    * Date - 51
    * Time - 61
    * Compound - 101

    Mapping from types to their numeric encodings was inferred from data.

    Categoricals
    ------------
    are represented by variety of concrete data types:
    * integers
    * strings
    * dates

    The required information is here:
    https://biobank.ndph.ox.ac.uk/showcase/schema.cgi (under
    "values for * encodings"). This implementation will use the following
    items to specify which columns should be represented as integers (other
    categoricals will encoded as strings).

    * "values for simple integer encodings"
    * "Values for hierarchical integer encodings"
    
    Parameters
    ----------
    csv_path : path to the UKB CSV instance to be converted
    n_chunks : how many parquet files to produce
    output_path : where the result should be stored (a directory path, wich 
      cannot exist and will be created).
    """
    output_path = Path(output_path)
    assert not output_path.exists(), f'Output folder already exists: {output_path}'
    output_path.mkdir()
    
    print('Fetching UKB schema...')
    ukb_schema = UkbCatalogue.schema
    
    print('Reading CSV header...')
    header = pd.read_csv(csv_path, nrows=0, index_col='eid')
    field_ids = {int(c.split('-')[0]) for c in header}

    schema_fields = set(ukb_schema.index)
    no_schema_fields = list(field_ids.difference(schema_fields))

    print('# distinct fields in data:', len(field_ids))
    print('# entries in the schema:', len(ukb_schema))
    print('Fields NOT included in schema:', no_schema_fields)
    print('Fields in schema but NOT in data:', 
          schema_fields.difference(field_ids))

    default_pandas_types = {
        11: 'float64',          # Integers are represented as floats for 
                                # performance reasons (it would have to be
                                # pd.Int64DType due to null values in data,
                                # which can be 2 orders of magnitude slower
                                # than float64 in some operations)
        21: pd.StringDtype(),   # Categorical (single)
        22: pd.StringDtype(),   # Categorical (multiple)
        31: 'float64',          # Continuous
        41: pd.StringDtype(),   # Text
        51: 'timestamp',        # Date
        61: 'timestamp',        # Time
        101: pd.StringDtype(),  # Compound
    }

    dtype_map = {}
    value_types = ukb_schema['value_type'].to_dict()

    print('Calculating dtype map.')
    for col in header.columns:
        field_id = int(col.split('-')[0])
        if field_id in schema_fields:
            dtype_map[col] = default_pandas_types[value_types[field_id]]

    # some corrections
    dtype_map['eid'] = 'int64'  # no nulls
    if '393-0.0' in dtype_map:
        del dtype_map['393-0.0']  # bad type annotation in UKB, useless data

    # pandas requires timestamp columns to be specified separately
    timestamp_cols = [col for col, dtype in dtype_map.items() if
                      dtype == 'timestamp']

    non_date_cols = {col: dtype for col, dtype in dtype_map.items()
                     if col not in timestamp_cols}

    col_names = ['eid'] + header.columns.to_list()

    print(f'Splitting CSV to {n_chunks} chunks.')
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        split_file(Path(csv_path), n_chunks, temp_dir)

        part_paths = [
            f for f in temp_dir.iterdir() if re.match(r'.*csv-split-\d+$', f.name)
        ]

        print('Starting dask cluster')
        client, cluster = spinup_dask_cluster(n_chunks)
        print('Started dask at:', client.dashboard_link)

        try:
            print('Converting chunks with Dask...')
            paths = client.compute([
                delayed(csv_to_parquet)(path,
                                        output_path,
                                        dtype_map,
                                        col_names,
                                        non_date_cols,
                                        timestamp_cols)
                for path in part_paths],
                sync=True)

            print('The following files have been created:')
            for p in paths:
                print('-', p)
        finally:
            client.close()
            cluster.close()
            if cleanup:
                remove_csv_parts(temp_dir)


def ehr_csv_to_parquet(csv_path, 
                        parquet_path=None, 
                        read_options=None, 
                        parse_options=None, 
                        convert_options=None,
                        outp_suffix='',
                        single_file=False,
                        column_names=None,
                        **pq_writer_args):
    """Converts an EHR (electronic health records) file (csv/txt) to parquet.
    A generic routine that splits the text file into chunks and rewrites them
    as parquet. Uses pyarrow CSV reader. All format specifics are configured
    via parameters.
    """
    csv_path = Path(csv_path)
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet' + outp_suffix)
    else:
        parquet_path = Path(parquet_path)

    print('Writing to', parquet_path)
    parquet_path.mkdir(exist_ok=True)
    
    def rename_cols(table):
        if column_names is not None:
            return table.rename_columns(column_names)
        else:
            return table
    
    if single_file:
        dest = parquet_path / f'part-0.parquet'
        table = csv.read_csv(
            csv_path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options)
        pq.write_table(rename_cols(table), dest, **pq_writer_args)
    else:
        reader = csv.open_csv(
            csv_path,
            read_options=read_options,
            parse_options=parse_options,
            convert_options=convert_options)
        for i, record_batch in tqdm(enumerate(reader)):
            table = pa.Table.from_batches([record_batch])
            dest = parquet_path / f'part-{i}.parquet'
            pq.write_table(rename_cols(table), dest, **pq_writer_args)
