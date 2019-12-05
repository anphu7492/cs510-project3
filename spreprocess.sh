#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=project_2001284
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=50G

source activate tensorflow2.0

echo "Start processing"

# python -u tokenization.py
python -u preprocess.py --model HAN \
                        --data_path data/HAN_1M

echo "End tokenization"
