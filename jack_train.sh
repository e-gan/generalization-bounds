#!/bin/bash
#SBATCH --job-name=bounds
#SBATCH --time=0-06:00:00
###SBATCH --gres=gpu:a100:1
###SBATCH --gres=gpu:RTXA6000:1
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
####SBATCH --mail-user=jackking@mit.edu
#SBATCH --mem=20G

source ~/.bashrc

module load openmind8/cuda/11.7
# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

SU_HOME="/om2/user/${USER_NAME}/generalization-bounds"

conda activate modular_transformers
echo $(which python)

python "${SU_HOME}/our_train.py"
