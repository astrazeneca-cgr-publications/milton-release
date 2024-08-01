import pandas as pd
import os
import json
import numpy as np
import sys
import random


def return_genes(code, 
                 associations_to_test_path, 
                 test_original, 
                 list_with_all_18k_genes_path):
    '''
    Function to return actual associations or random set of genes of same length for a given ICD10 code
    Input:
        code: ICD10 code (for example 'N18')
        associations_to_test_path: CSV file containing 3-character ICD10 codes in column ('ICD10_CODE_3_char') and associated gene in column 'gene' (one associated gene per row).
        test_original: True if associated gene(s) list is to be returned, False if randomly selected genes of same length are to be returned.
        list_with_all_18k_genes_path: parquet file containing all the genes that can be used for sampling randomly selected genes in a column 'gene'.
    Output: gene list containing actual associated genes if test_original ==True or randomly selected genes if test_original==False
    '''
    hits=pd.read_csv(associations_to_test_path)
    original_genes=hits.loc[hits.ICD10_CODE_3_char==code, 'gene'].unique().tolist()
    
    if test_original:
        return original_genes
    else:
        tmp=pd.read_parquet(list_with_all_18k_genes_path).reset_index() #read file containing ~18k genes
        return random.sample(tmp.gene.unique().tolist(), len(original_genes))

def run_lookup(code, associations_to_test_path, test_original, finngen_res_path, mappings_path, list_with_all_18k_genes_path, out_file):
    '''
    Input:
        finngen_res_path : path to FinnGen GWAS summary statistics files
        associations_to_test_path: CSV file containing 3-character ICD10 codes in column ('ICD10_CODE_3_char') and associated gene in column 'gene' (one associated gene per row).
        mappings_path: path to CSV file containing ICD10 codes as one column (ICD10) and mapped FinnGen phenotype as another column (NAME). 
        test_original: True if associated gene(s) list is to be returned, False if randomly selected genes of same length are to be returned.
        out_file : full file name where to save results
    Output: saves retrieved rows from FinnGen summary stats file into a parquet file or prints a message no row is retrieved.
    '''
    #depending on test_original, extract a list of genes to lookup in FinnGen
    genes=return_genes(code, associations_to_test_path, test_original, list_with_all_18k_genes_path)
    
    #define data frame to save results
    all_res=pd.DataFrame()
    
    #extract FinnGen phenotypes mapped to this code.
    all_mappings=pd.read_csv(mappings_path)
    names=all_mappings.loc[all_mappings.ICD10==code, 'NAME'].unique().tolist()
        
    for name in names:
        filename=os.path.join(finngen_res_path, name+'.gz')
        if os.path.exists(filename): #proceed if file exists
            tmp=pd.read_csv(filename, compression='gzip', header=0, sep='\t', quotechar='"') #read file
            sub=tmp[tmp.pval<0.05] #subset to only those that have p<0.05
            split_cols=sub.nearest_genes.str.split(',', expand=True) #expand nearest genes column because there can be multiple gene entries per row.
            
            for gene in genes:
                lookup=(split_cols==gene).any(axis=1) #check if a gene exists
                all_res=pd.concat([all_res, sub.loc[lookup[lookup==True].index, :].assign(NAME=name, ICD10=code, gene_looked_up=gene, tested_original=test_original)]) #extract rows containing this gene entry and concatanate results to final dataframe. 'gene_looked_up' column is added to keep track of which gene was tested in case there are multiple gene entries per row to avoid confusion later.
            
    #save to file if there is at least one row
    if len(all_res)>0:
        all_res.to_parquet(out_file)
    else:
        print('Empty DF returned!')
    
def function_to_run(code, associations_to_test_path, test_original, finngen_res_path, mappings_path, list_with_all_18k_genes_path, out_dir):
    '''
    Master function to 
    Function to lookup given variant-ICD10 association in respective FinnGen phenotype files and retrieve all rows where given variant is associated with a given FinnGen phenotype (p<0.05).
    Input: 
        code: ICD10 code (for example, 'N18')
        associations_to_test_path: CSV file containing 3-character ICD10 codes in column ('ICD10_CODE_3_char') and associated gene in column 'gene' (one associated gene per row).
        test_original: True if associated gene(s) list is to be returned, False if randomly selected genes of same length are to be returned.
        finngen_res_path : path to FinnGen GWAS summary statistics files
        mappings_path: path to CSV file containing ICD10 codes as one column (ICD10) and mapped FinnGen phenotype as another column (NAME). 
        out_dir: path to output directory.
    
    Output: calls the function run_lookup, otherwise a printed message if file already exists.
    '''
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_file=os.path.join(out_dir, code+'.parquet')
    if not os.path.exists(out_file):
        print('{} does not exist yet'.format(out_file))
        '''This is the job that you want to submit that is the same for all the HPs, just changes its value '''
        run_lookup(code, associations_to_test_path, test_original, finngen_res_path, mappings_path, list_with_all_18k_genes_path, out_file)
    else:
        print('Result already exist')

def runmymodel(job):
    '''
    Master function to parse all necessary variables and call relevant functions
    Input: 
        job: sbatch array ID such as 1, 2, 3, 4..
    '''
    params = pd.read_csv('./sbatch_combinations.csv')
    current_param = params.iloc[int(job)-1]
    code = current_param.code
    associations_to_test_path=current_param.associations_to_test_path
    test_original=current_param.test_original
    finngen_res_path=current_param.finngen_res_path
    mappings_path=current_param.mappings_path
    list_with_all_18k_genes_path=current_param.list_with_all_18k_genes_path
    out_dir = current_param.out_dir
    function_to_run(code, associations_to_test_path, test_original, finngen_res_path, mappings_path, list_with_all_18k_genes_path, out_dir)

def main(job):
    runmymodel(job)

if __name__ == "__main__":
    main(sys.argv[1])