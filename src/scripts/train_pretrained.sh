#!/bin/bash

pretrain_name=$1
model_config=$2
pt_config=$3
OUTPATH=$4
seed=$5
echo $pretrain_name
echo $model_config
echo $pt_config



python src/model/train_LM.py -o ${OUTPATH}/${pretrain_name}/$pretrain_name -p $pretrain_name -c True -P True --model_config_path $model_config --pretrain_config_path $pt_config --load_checkpoint True --data_seed $seed