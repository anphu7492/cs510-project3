#!/bin/bash

#SBATCH -J train
#SBATCH --account=project_2001284
#SBATCH --mem-per-cpu 50000
#SBATCH --gres=gpu:v100:2
#SBATCH -p gpu
#SBATCH -t 4:00:00

# module load cuda/10.1.168 ???
source activate tensorflow2.0

echo "Start training..."
# baseline, TextAttBiRNN, HAN
python -u train_and_test.py --model baseline \
                            --save_dir baseline-clean-1M-cont \
                            --data_path data/baseline_1M \
                            --train_from output/baseline-clean-1M-es_122805/model-09-0.92965.hdf5

echo "Training done..."