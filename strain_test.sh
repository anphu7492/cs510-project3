#!/bin/bash

#SBATCH -J train
#SBATCH --mem-per-cpu 50000
#SBATCH --gres=gpu:v100:1
#SBATCH -p gpu
#SBATCH -t 3:00:00
##SBATCH --begin=02:00

source /appl/soft/ai/miniconda3/etc/profile.d/conda.sh
source activate tensorflow2.0

echo "Start training..."

python -u train_and_test.py --model TextAttBiRNN \
                            --save_dir att-birnn

echo "Training done..."