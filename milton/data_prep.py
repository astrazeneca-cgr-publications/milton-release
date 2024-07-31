"""Collection of routines for transformations of raw UKB datasets.
The transformations include:
- CSV to parquet conversions for standard UKB tables
- genomic matrices in CSV format to sparse matrix format
- malignant neoplasm survival dataset
"""

import pandas as pd
import numpy as np
import re
from os import path
import scipy.sparse as sp
from pathlib import Path
from pyarrow import csv
import pickle
from tqdm import tqdm
import logging
import gc


def csv_matrix_to_sparse(path, 
                         out_path=None, 
                         transpose=False, 
                         block_size_mb=200,
                         index_col='sample/gene',
                         delimiter='\t', 
                         na_repr=-1):
    """Utility function for converting genotype matrices as CSV into 
    the sparse matrix format. Although the function processes the 
    files in batches, it allocates large amounts of swap (around 
    20G per file). Use the block_size_mb argument to make the blocks
    smaller if memory use were an issue.
    
    Parameters
    ----------
    out_path : path to the file to be written
    transpose : if True, the result is transposed before writing. Regardless
        of the value, the result is stored as a CSR sparse matrix, meaning
        that filtering of rows is cheap.
    block_size_mb : number of MB of input CSV to process in one chunk. Less 
        chunks means faster processing but larger memory use.
    index_col : name of the index column 
    delimiter : the CSV delimiter character to use
    na_repr : integer representation for null values (if any)
    
    Returns
    -------
    index, columns, matrix - in order of appearance: the CSV's index as 
        numpy array, the CSV's column names as numpy array, the CSV data
        as scipy's CSR matrix.
    """
    path = Path(path)
    it = csv.open_csv(
        path,
        read_options=csv.ReadOptions(block_size=block_size_mb * 2**20), 
        parse_options=csv.ParseOptions(delimiter=delimiter))
    
    columns = None
    index = []
    parts = []
    
    for batch in tqdm(it):
        df = batch.to_pandas().drop('', axis=1, errors='ignore')
        # expecting the first column to be the index (gene names)
        df = df.set_index(df.columns[0]).fillna(na_repr).astype('uint8')
        
        if columns is None:
            try:
                int(df.columns[0])   # check if columns are numeric
                columns = df.columns.astype('int').to_numpy(copy=True)
            except ValueError:
                columns = df.columns.to_numpy(copy=True)
            
        index.append(df.index.to_numpy(copy=True))
        parts.append(sp.csr_matrix(df))

        # free up some of the memory
        del df
        for gen in [0, 1, 2]:
            gc.collect(gen)
    
    index = np.concatenate(index)
    matrix = sp.vstack(parts)
    
    if transpose:
        # make it a CSR matrix (good for row slicing)
        # with patient ids in rows
        matrix = matrix.T.tocsr()
        tmp = index
        index = columns
        columns = tmp
        
    result = index, columns, matrix
    
    try:
        if not out_path:
            out_path = path.with_suffix('.pickle')
        else:
            out_path = Path(out_path)
            if out_path.is_dir():
                out_path = out_path / path.with_suffix('.pickle').name
        with out_path.open('wb') as f:
            pickle.dump(result, f)
    except Exception as ex:
        logging.error(ex)

    return result
