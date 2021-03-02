#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --time=01:00:00

module load miniconda/3
conda activate ift-hw1

python gold_q5_vanilla.py


