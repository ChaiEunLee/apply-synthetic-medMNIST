#!/bin/bash

#SBATCH --job-name=deit                    # Submit a job named "example"
#SBATCH --nodes=1                           #Using 1 node --nodelist=b19
#SBATCH --gres=gpu:1                          # Using 1 gpu
#SBATCH --time=0-04:00:00                     # 1 hour timelimit
#SBATCH --mem=10000MB                         # Using 10GB CPU Memory
#SBATCH --partition=P2                         # Using "b" partition 
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor

#source ${HOME}/.bashrc
source ${HOME}/anaconda/bin/activate
conda activate transfer

python ${HOME}/Transfer/experiment/Deit_synthetic.py
