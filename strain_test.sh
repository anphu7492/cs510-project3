#!/bin/bash

#SBATCH -J train
#SBATCH --account=project_2001284
#SBATCH --mem-per-cpu 50000
#SBATCH --gres=gpu:v100:2
#SBATCH -p gpu
#SBATCH -t 7:00:00

# module load cuda/10.1.168 ???
source activate tensorflow2.0

echo "Start training..."
# baseline, TextAttBiRNN, HAN
python -u train_and_test.py --model Regularized \
                            --save_dir Regularized-clean-1M-es \
                            --data_path data/baseline_1M

echo "Training done..."