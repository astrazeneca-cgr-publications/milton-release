import pandas as pd
import os
import pickle
import numpy as np
import sys
import random

def load_annotations(path_to_ExWAS_associations_pickle_file):
    '''
    Input: 
        path_to_ExWAS_associations_pickle_file: path to pickle file containing ExWAS associations in a dictionary data type with keys as annotation 'baseline'/'putative novel'/'known quant' and values as list of (variant, ICD10-code) tuples
        
    Output: variable to which this loaded dictionary was assigned
    '''
    with open(path_to_ExWAS_associations_pickle_file, 'rb') as f:
        annotated=pickle.load(f)
    return annotated

def run_lookup(code, path_to_zipped_FinnGen_summary_stats_dir, out_path, names, annotated):
    '''
    Function to lookup given variant-ICD10 association in respective FinnGen phenotype files and retrieve all rows where given variant is associated with a given FinnGen phenotype (p<0.05).
    Input: 
        code: ICD10 code (for example, 'N18')
        path_to_zipped_FinnGen_summary_stats_dir: path to directory containing FinnGen GWAS summary statistics files.
        out_path: name of parquet file in which to save retrieved rows.
        names: FinnGen phenotypes mapped to given ICD10 code
        annotated: dictionary with annotation as keys and a list of tuples containing variant-ICD10 associations as values.
    
    Output: parquet file with retreived associations from FinnGen (otherwise a printed message if no row is retrieved)
    '''
    all_res=pd.DataFrame()
    for name in names:
        filename=os.path.join(path_to_zipped_FinnGen_summary_stats_dir, name+'.gz')
        if os.path.exists(filename): #if file exists do the following
            tmp=pd.read_csv(filename, compression='gzip', header=0, sep='\t', quotechar='"') #read FinnGen summary statistics
            #fetch p-values corresponding to the genotype
            tmp_sub=tmp[tmp.pval<0.05].copy() #subset to only significant FinnGen associations for faster downstream processing
            tmp_sub.loc[:, 'genotype']=tmp_sub['#chrom'].astype('str')+'-'+tmp_sub['pos'].astype('str')+'-'+tmp_sub['ref']+'-'+tmp_sub['alt']
            tmp_sub.loc[:, 'genotype_flipped']=tmp_sub['#chrom'].astype('str')+'-'+tmp_sub['pos'].astype('str')+'-'+tmp_sub['alt']+'-'+tmp_sub['ref'] #generating alternate versions because sometimes reference and alternate alleles are flipped between UKB and FinnGen
            for key in ['baseline', 'putative novel']:
                sub=pd.DataFrame(annotated[key], columns=['genotype', 'phenotype']) #extracting information from dictionary instead of full dataframe
                goi=sub[sub.phenotype==code].genotype.unique()
                if len(goi)>=1: #if there is at least 1 gene available, do this
                    res=tmp_sub.loc[(tmp_sub.genotype.isin(goi))|(tmp_sub.genotype_flipped.isin(goi))] #check if the genotype in FinnGen match the default or flipped genotype in UKB
                    all_res=pd.concat([all_res, res.assign(phenotype=code).assign(NAME=name).assign(annotation=key)])

    #save 1 file per code if at least 1 row exists in the file
    if all_res.shape[0]>=1:
        all_res.to_parquet(out_path)
    else:
        print('Empty DF returned!')
    
def function_to_run(code, path_to_ICD10_FinnGenPhenotype_mappings_file, path_to_ExWAS_associations_pickle_file, path_to_zipped_FinnGen_summary_stats_dir, out_dir):
    '''
    Function to make relevant directory and prepare all inputs for run_lookup function.
    Input: 
        code: ICD10 code (for example, 'N18')
        path_to_ICD10_FinnGenPhenotype_mappings_file: parquet file containing ICD10 codes as one column (COD_ICD_10) and mapped FinnGen phenotype as another column (NAME). 
        path_to_ExWAS_associations_pickle_file: pickle file containing a dictionary with keys as annotation 'baseline'/'putative novel'/'known quant' and values as list of (variant, ICD10-code) tuples
        path_to_zipped_FinnGen_summary_stats_dir: path to directory containing FinnGen GWAS summary statistics files.
        out_dir: path to output directory.
    
    Output: calls the function run_lookup, otherwise a printed message if file already exists.
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_path=os.path.join(out_dir, code+'.parquet')
    #if file already does not exist
    if not os.path.exists(out_path):
        print('{} does not exist yet'.format(out_path))
        '''This is the job that you want to submit that is the same for all the HPs, just changes its value '''
        
        #load ICD10 to FinnGen mappings
        mappings=pd.read_parquet(path_to_ICD10_FinnGenPhenotype_mappings_file)
        names=mappings[mappings.COD_ICD_10==code].NAME.to_list() #there could be multuple phenotypes mapped ot the same ICD10 code
    
        #load annotations
        annotated=load_annotations(path_to_ExWAS_associations_pickle_file)
        
        #run the function
        run_lookup(code, path_to_zipped_FinnGen_summary_stats_dir, out_path, names, annotated)
    else:
        print('Result already exist')

def runmymodel(job):
    '''
    Master function to parse all necessary variables and call relevant functions
    Input: 
        job: sbatch array ID such as 1, 2, 3, 4..
    '''
    #load the file containing all parameters
    params = pd.read_csv('./sbatch_combinations.csv')
    current_param = params.iloc[int(job)-1]
    #assign to variables
    code = current_param.ICD10_CODE #For example, string 'N181' for ICD10 code N18.1
    out_dir = current_param.out_dir #directory for saving output results
    path_to_ICD10_FinnGenPhenotype_mappings_file = current_param.mappings_dir #file containing ICD10 code to FinnGen phenotype mappings
    path_to_ExWAS_associations_pickle_file = current_param.EXWAS_associations_file 
    path_to_zipped_FinnGen_summary_stats_dir = current_param.FinnGen_summary_stats_dir #zipped publicly available FinnGen GWAS summary statistics file in format <phenotype>.gz
    #call relevant function
    function_to_run(code, path_to_ICD10_FinnGenPhenotype_mappings_file, path_to_ExWAS_associations_pickle_file, path_to_zipped_FinnGen_summary_stats_dir, out_dir)

def main(job):
    runmymodel(job)

if __name__ == "__main__":
    main(sys.argv[1])