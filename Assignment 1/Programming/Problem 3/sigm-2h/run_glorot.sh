#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=12G
#SBATCH --time=00:20:00

module load miniconda/3
conda activate ift-hw1

python main_glorot.py


