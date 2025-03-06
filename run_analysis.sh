#!/bin/bash

#SBATCH --job-name=analysis
#SBATCH --partition=sleuths
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=10:00:00

#SBATCH --output=/home/kong/out_sbatch/sparsify/cifar/analysis/run%j.txt
#SBATCH --error=/home/kong/out_sbatch/sparsify/cifar/analysis/err_run%j.txt

echo "Running on $(hostname)"
python all_analysis.py
echo "Job finished at $(date)"
