#!/bin/bash

MODEL_DIR_PATH=$1
MODEL_CONFIG_PATH=$2
EVAL_DATA=$3
OUT=$4
echo model path: $MODEL_DIR_PATH
echo model config path: $MODEL_CONFIG_PATH
echo data used for eval: $EVAL_DATA
echo

python eval/eval.py -e $EVAL_DATA -m $MODEL_DIR_PATH -c  $MODEL_CONFIG_PATH -o $OUT
