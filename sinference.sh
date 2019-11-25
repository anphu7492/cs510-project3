#!/bin/bash

echo "Starting inference..."

python -u inference.py  --model TextAttBiRNN \
                        --checkpoint data/models/TextAttBiRNN/112419_04/model-01-0.68452.hdf5

echo "Inference done...."
