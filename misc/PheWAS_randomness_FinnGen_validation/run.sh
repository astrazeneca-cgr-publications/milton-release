#!/bin/bash
#SBATCH -t 2:00:00
#SBATCH --mem=16G
#SBATCH --partition=gpu,core
#SBATCH --cpus-per-task 8
#SBATCH --nodes 1
#SBATCH -J slurm_%A_%a
#SBATCH --array=1-2255%100

module load Miniconda3/4.12.0
source activate milton #activate conda environment 
python3 ./runmymodel.py $SLURM_ARRAY_TASK_ID
