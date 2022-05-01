#!/bin/bash

pt_name=$1
ft_name=$2
pt_config=$3
ft_config=$4
model_config=$5
OUTPATH=$6
seed=$7

echo $pt_name
echo $model_config
echo $ft_config
echo $is_mapped


python src/model/train_LM.py -o ${OUTPATH}/$pt_name -p $pt_name -f $ft_name -c True -m False --model_config_path $model_config --finetune_config_path ${ft_config} --pretrain_config_path ${pt_config} --load_checkpoint True --data_seed ${seed}