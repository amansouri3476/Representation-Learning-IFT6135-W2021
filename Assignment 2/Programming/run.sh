#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --time=06:00:00
#SBATCH -o out/%j.out
#SBATCH -e err/%j.err
#SBATCH --partition=long

module load miniconda/3
conda activate ift-hw2

python run_exp.py --model=${MODEL} --layers=${LAYERS} --batch_size=${BATCH_SIZE} --epochs=${EPOCHS} --optimizer=${OPTIMIZER} --exp_id=${EXP_ID}

conda deactivate
module purge
