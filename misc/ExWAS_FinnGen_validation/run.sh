#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu,core
#SBATCH --cpus-per-task 8
#SBATCH --nodes 1
#SBATCH -J slurm_%A_%a
#SBATCH --array=1-487%100

module load Miniconda3/4.12.0
source activate milton #activate the conda environment
python3 ./runmymodel.py $SLURM_ARRAY_TASK_ID #can also provide full path: <path_to_dir>/.conda/envs/milton/bin/python3
