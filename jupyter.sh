#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --partition=sleuths
#SBATCH --reservation=kaschube-shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=8:00:00

#SBATCH --output=/home/kong/out_sbatch/sparsify/jupyter/run%j.txt
#SBATCH --error=/home/kong/out_sbatch/sparsify/jupyter/err_run%j.txt



cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8892

