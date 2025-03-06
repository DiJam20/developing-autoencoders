#!/bin/bash

#SBATCH --job-name=cnn
#SBATCH --partition=sleuths
#SBATCH --reservation=kaschube-shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=30:00:00

#SBATCH --output=/home/kong/out_sbatch/sparsify/cifar/run%j.txt
#SBATCH --error=/home/kong/out_sbatch/sparsify/cifar/err_run%j.txt

echo "Running on $(hostname)"
python cifar_cnn.py
echo "Job finished at $(date)"
