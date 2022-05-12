<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [The Linguistic Blood Bank](#the-linguistic-blood-bank)
  - [Interactive-Exploration](#interactive-exploration)
  - [Reproducing our results](#reproducing-our-results)
    - [Installation and Requirements](#installation-and-requirements)
  - [Reproducing Evaluations with Pretrained Models](#reproducing-evaluations-with-pretrained-models)
  - [Training from Scratch](#training-from-scratch)
    - [Tokenizer](#tokenizer)
    - [Base model](#base-model)
    - [Finetuned model on top of an existing one](#finetuned-model-on-top-of-an-existing-one)
    - [Evaluate MRR](#evaluate-mrr)
  - [Downstream training](#downstream-training)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# The Linguistic Blood Bank
Supporting repo for
[Balanced Data Approach for Evaluating Cross-Lingual Transfer: Mapping
the Linguistic Blood Bank](https://arxiv.org/abs/2205.04086) <br>
_Dan Malkin, [Tomasz Limisiewicz](https://tomlimi.github.io/), [Gabriel Stanovsky](https://gabrielstanovsky.github.io/)_, <br>
NAACL 2022.

## Interactive-Exploration

<b>To view and interact with our results:</b> visit [this url](https://share.streamlit.io/dnmh/language-graph/code_cleanup/visualization_tool/launch_interface.py). 

If you want to deploy this server locally, see [instructions here](running_sever_locally.md)


## Reproducing our results

### Installation and Requirements

from the root directory run:
```
pip install -r requirements.txt
```

Tested on python 3.7.

## Reproducing Evaluations with Pretrained Models

* Download the processed wikipedia data at https://drive.google.com/file/d/1q5eOxc-cNT1YXV2eVG8jqZBLsPEQ2_Ld/view?usp=sharing and unpack to a desired directory.
* To get the information approximation we use in our work, first save all tokens of a wanted corpus into a file TOKENS_FILE. This should be done by simply running the tokenizer on the data's line, and writing each token to a file on it's own line. Then run the following from the project's root directory:
```
python data/info_analysis.py -t TOKENS_FILE
```
* This returns the total number of tokens, followed by unique number of tokens and the ratio of both.
An example of such processed tokens file is:
```
he
he
llo
llo
world
world
```
* Running the script with this file will output: 
```
python data/info_analysis.py -t data/example_files/example_tokens.txt
...
INFO:root:total tokens:6, unique tokens:3, ratio:0.5
```

## Training from Scratch

*Before you start: training configurations*

In order to run a training you must create a training configuration which includes the paths to the training data, model parameters, and training procedure parameters.
Each run get specification via a model config and a training procedure config. To create a config run the following:
```
python src/model/train_utils.py ARGS
```

for example, to create a model config and training config to produce a monolingual english model with 6 hidden layer and 8 attention heads run:
```
python src/model/train_utils.py -o ~/LMs/en -c model pretrain --pt_train_data_list DATA_PATH/en/train.txt --pt_eval_data_list DATA_PATH/en/test.txt --pt_num_epochs 3 --pt_batch_size 8 --pt_name pt_config --tokenizer_path TOKENIZER_PATH --vocab_size 100000 --hidden_size 512 --max_len 128 --num_attention 8 --num_hidden 6 --model_name en_model
```
this will output pt_config.json and en_model.json into ~/LMs/en
where pt_config.json contains:
```
{"train_data_paths_list": ["DATA_PATH/en/train.txt"], 
"eval_data_paths_list": ["DATA_PATH/en/test.txt"], 
"num_epochs": 3, 
"batch_size": 8}
```
and en_model.json contains:
```
{"tokenizer_path": "TOKENIZER_PATH", "hidden_layer_size": 512, "vocab_size": 100000, "max_sent_len": 128, "num_hidden": 6, "num_attention": 8}
```

the ARGS are specified as followed:
Specify to config type to produce (pretrain config, finetune config, model parameters config)
```
    '-c','--config_types', nargs='+', help='a list from {'pretrain','finetune','model'}
```

Paths to training data lists (finetune or pretrain) as well as the output path under which all configs will be generated:
```
    '-o','--out_path', type=str, 
    '--pt_train_data_list', nargs='+'
    '--pt_eval_data_list', nargs='+'
    '--ft_train_data_list', nargs='+'
    '--ft_eval_data_list', nargs='+'
```
Arguments to define the training pipeline and model parameters:
```
    '--pt_num_epochs',type=int
    '--pt_batch_size',type=int
    '--ft_num_epochs',type=int
    '--ft_batch_size',type=int
    '--tokenizer_path',type=str
    '--vocab_size',type=int
    '--hidden_size',type=int
    '--max_len',type=int
    '--num_hidden',type=int
    '--num_attention',type=int
```
Arguments to define the training pipeline and model parameters (the names of the config files, will output: model_name.json, pt_name.json, ft_name.json):
```
    '--model_name',type=str
    '--pt_name',type=str
    '--ft_name',type=str
```
### Tokenizer
train a BertWordPieceTokenizer from the tokenizers library (transformers by Huggingface) on your desired data. All desired languages should be included here. 

### Base model 
to train a base model run the following script from your root directory:
```
bash src/scripts/train_pretrained.sh PRETRAIN_SOURCE_NAME MODEL_CONFIG_PATH PRETRAIN_CONFIG_PATH OUTPUT_DIR SEED
```
This will output a model specified by  MODEL_CONFIG_PATH & PRETRAIN_CONFIG_PATH into OUTPUT_DIR/PRETRAIN_SOURCE_NAME/PRETRAIN_SOURCE_NAME_SEED

### Finetuned model on top of an existing one
To train a finetuned mlm model on top of the previously trained model in OUTPUT_DIR/PRETRAIN_SOURCE_NAME/PRETRAIN_MODEL_NAME_SEED, run the following from your root directory:
```
bash src/scripts/train_finetuned.sh PRETRAIN_SOURCE_NAME FINETUNED_TARGET_NAME PRETRAIN_CONFIG_PATH FINETUNE_CONFIG_PATH MODEL_CONFIG_PATH OUTPUT_DIR SEED
```
Make sure PRETRAIN_CONFIG_PATH, OUTPUT_DIR and MODEL_CONFIG_PATH are identical to what you ran in the base model.
This will output the model at UTPUT_DIR/PRETRAIN_SOURCE_NAME/PRETRAIN_SOURCE_NAME_SEED while finetuned on the data specified by FINETUNE_CONFIG_PATH into OUTPUT_DIR/PRETRAIN_SOURCE_NAME/FINETUNED_TARGET_NAME_SEED

Example: train an arabic model on top of a russian monolingual model:
```
bash src/scripts/train_pretrained.sh ru MODEL_CONFIG_PATH PRETRAIN_CONFIG_PATH OUTPUT_DIR 10 >> OUTPUT_DIR/ru/ru_10/*
bash src/scripts/train_finetuned.sh ru ar PRETRAIN_CONFIG_PATH FINETUNE_CONFIG_PATH MODEL_CONFIG_PATH OUTPUT_DIR 10 >> OUTPUT_DIR/ru/ar_10/* 
```

### Evaluate MRR
Given a model saved in MODEL_DIR_PATH and defined by the mentioned config file at MODEL_CONFIG_PATH, to evaluate its performance on a given data EVAL_DATA and save the data in OUTPUT_DIR_PATH, run the following script from the root directory:
```
bash src/scripts/eval_model_mrr.sh MODEL_DIR_PATH MODEL_CONFIG_PATH EVAL_DATA OUTPUT_DIR_PATH
```

## Downstream training

To train the downstream task ontop of a given langauge model, follow the instructions at: https://github.com/google-research/xtreme .

To truncate the data as we did, limit the data in the files outputed by https://github.com/google-research/xtreme/blob/master/utils_preprocess.py to the desired example number (2000 for POS, 5000 for NER). This can be done manually by truncating the files or by changing the preprocess function to stop after X examples. 

Make sure to additionally preprocess the non-english training files as well (the default behaviour process only english train files) using the same code. 

