#!/bin/bash
#SBATCH --job-name=bounds
#SBATCH --time=1-00:00:00
###SBATCH --gres=gpu:a100:1
###SBATCH --gres=gpu:RTXA6000:1
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
####SBATCH --mail-user=jackking@mit.edu
#SBATCH --mem=60G

source ~/.bashrc

module load openmind8/cuda/11.7
# find the user name
USER_NAME=$(whoami)
unset CUDA_VISIBLE_DEVICES

SU_HOME="/om2/user/${USER_NAME}/generalization-bounds"

conda activate modular_transformers
echo $(which python)

###python "${SU_HOME}/our_train.py" -m training.learning_rate=0.01,0.1 training.lr_scheduler_params.gamma=0.95,0.98 data.corruption_type=random_labels,None

##python "${SU_HOME}/our_train.py" training.learning_rate=0.1 training.lr_scheduler=StepLR data.corruption_type=None training.batch_size=64 model.name=AlexNet training.num_epochs=400 training.lr_scheduler_params.gamma=0.95

python "${SU_HOME}/our_train.py" ###-m data.corruption_prob=0.1,0.25,0.5,0.75,1.0

python "${SU_HOME}/our_train.py" data.corruption_type=None