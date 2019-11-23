#!/bin/bash
FILE=${1:"test_run.out"}
python -u train_and_test.py 2>&1 | tee $FILE


