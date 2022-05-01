import torch
import argparse
from transformers import BertTokenizer, set_seed
from transformers import BertConfig, BertForMaskedLM # , DataCollatorForLanguageModeling
import logging
import sys
import os, pickle
from transformers import AutoModelForMaskedLM
import json
from dataset import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

set_seed(10)
logging.info(torch.cuda.is_available())

def pretrain_and_finetune(pretrain_outpath, ft_output_path, mapping_model, model_config, pt_config, ft_config, eval_only,pretrain_only, truncate_at, load_checkpoint=True, data_seed=10):
    logging.info("Loading tokenizer..")
    # get tokenizer:
    tokenizer = BertTokenizer.from_pretrained(model_config['tokenizer_path'])
    MASK_ID = tokenizer.mask_token_id
    logging.info(MASK_ID)
    # build dataset:
    logging.info("Building Model..")
    config = BertConfig(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_layer_size'],
        num_hidden_layers=model_config['num_hidden'],
        num_attention_heads=model_config['num_attention'],
        max_position_embeddings=model_config['max_sent_len']
    )
    model = BertForMaskedLM(config)
    # define a data_collator (a small helper that will help us batch different samples of the dataset together into an
    # object that PyTorch knows how to perform backprop on):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    # init trainer:
    logging.info("Training\Loading model...")
    logging.info("#params:,",model.num_parameters())
    if not os.path.exists(os.path.join(pretrain_outpath,'config.json')):
        logging.info("Loading pretrain data..")
        pretrain_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_paths= pt_config['train_data_paths_list'], block_size=model_config['max_sent_len'], truncate_at=truncate_at, name="pretrain train", rand_seed=data_seed)
        preeval_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_paths=pt_config['eval_data_paths_list'], block_size=model_config['max_sent_len'], truncate_at=truncate_at, name="pretrain eval", rand_seed=data_seed)
        logging.info("Pretraining model..")
        os.makedirs(pretrain_outpath, exist_ok=True)
        training_args = TrainingArguments(
            output_dir=pretrain_outpath,
            overwrite_output_dir=True,
            num_train_epochs=pt_config['num_epochs'],
            per_device_train_batch_size=pt_config['batch_size'],
            save_steps=1000,
            save_total_limit=4,
            report_to='none',
        )
        logging.info("Reporting to: ", training_args.report_to)
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=pretrain_dataset,
            eval_dataset=preeval_dataset,
        )
        if load_checkpoint:
            logging.info("loading pt checkpoint")
            try:
                trainer.train(resume_from_checkpoint=True)
            except Exception as e:
                logging.info("Failed loading checkpoint, regular training")
                trainer.train()
        else:
            logging.info("training pt from scratch")
            trainer.train()
        trainer.save_model(pretrain_outpath)
        logging.info("Done pretrain. pretrained model saved in:\n",pretrain_outpath)
        metrics = trainer.evaluate()
        logging.info(metrics)
        with open(os.path.join(pretrain_outpath,'pretrain_eval.pickle'), 'wb') as evalout:
            pickle.dump(metrics, evalout, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        logging.info("pt model exists, loading model from ", pretrain_outpath)
        model = AutoModelForMaskedLM.from_pretrained(pretrain_outpath)
    if pretrain_only:
        return
    assert ft_output_path
    if ft_output_path is None:
        with open(sys.argv[0], 'r') as model_code, open(os.path.join(pretrain_outpath,'pretrain_source_code.py'), 'w') as source_out :
            code_lines = model_code.readlines()
            source_out.writelines(code_lines)
        logging.info("Only pretrain, Done.")
        return

    logging.info("Loading ft data..")
    ft_train = LineByLineTextDataset(tokenizer=tokenizer, file_paths= ft_config['train_data_paths_list'], block_size=model_config['max_sent_len'],truncate_at=truncate_at, name="ft train", rand_seed=data_seed)
    ft_eval = LineByLineTextDataset(tokenizer=tokenizer, file_paths= ft_config['eval_data_paths_list'], block_size=model_config['max_sent_len'],truncate_at=truncate_at, name="ft eval", rand_seed=data_seed)
    training_args = TrainingArguments(
        output_dir=ft_output_path,
        overwrite_output_dir=True,
        num_train_epochs=ft_config['num_epochs'],
        per_device_train_batch_size=ft_config['batch_size'],
        save_total_limit=4,
        report_to='none',
    )
    logging.info("Reporting to: ", training_args.report_to)
    ft_trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=ft_train,
        eval_dataset=ft_eval,
    )
    if not eval_only:
        logging.info("Finetuning model..")
        if mapping_model:
            logging.info("Freezing model..")
            # freeze bert execpt mlm head and encoding layer.
            for param in model.bert.parameters():
                param.requires_grad = False
            for param in model.bert.embeddings.parameters():
                param.requires_grad = True
            ft_trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=ft_train,
                    eval_dataset=ft_eval,
                )
        if load_checkpoint:
            logging.info("loading ft checkpoint")
            try:
                ft_trainer.train(resume_from_checkpoint=True)
            except Exception as e:
                logging.info("Failed loading checkpoint, regular ft training")
                ft_trainer.train()
        else:
            logging.info("training ft from scratch")
            ft_trainer.train()
        ft_trainer.save_model(ft_output_path)
        logging.info("Done ft. Model saved in:\n",ft_output_path)
    else:
        logging.info("Only eval, no training for ft.")
        model = AutoModelForMaskedLM.from_pretrained(ft_output_path)
        ft_trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=ft_train,
            eval_dataset=ft_eval,
        )
        logging.info("ft. Model loaded from:\n",ft_output_path)
    logging.info("Evaluating model...")
    metrics = ft_trainer.evaluate()
    logging.info(metrics)
    with open(os.path.join(ft_output_path,'eval.pickle'), 'wb') as evalout:
        pickle.dump(metrics, evalout, protocol=pickle.HIGHEST_PROTOCOL)
    with open(sys.argv[0], 'r') as model_code, open(os.path.join(ft_output_path,'source_code.py'), 'w') as source_out :
        code_lines = model_code.readlines()
        source_out.writelines(code_lines)


def load_config(config_path):
    with open(config_path, 'r') as fp:
        return json.load(fp)


def train(args):
    out_path = args.out
    is_common_pretrain = args.is_common_pretrain
    pretrain_name = args.pretrain_name
    ft_name = args.ft_name
    data_seed = args.data_seed
    mapping_model = False
    logging.info('Training mulitling')
    if is_common_pretrain:
        pretrain_outpath = os.path.join(out_path, pretrain_name+'_'+str(data_seed))
        logging.info("Common pretrain path: ",pretrain_outpath)
    else:
        assert ft_name, 'pretrain is unique to finetune target but no target (ft) name was given'
        pretrain_outpath = os.path.join(out_path, pretrain_name,ft_name +'_'+str(data_seed))
        logging.info("pretrain path: ",pretrain_outpath)
    if args.pretrain_only:
        logging.info("Only pretraining.")
        ft_output_path = None
    elif not mapping_model:
        assert ft_name,'has finetune target but no target (ft) name was given'
        ft_output_path = os.path.join(out_path, ft_name+'_'+str(data_seed))
    else:
        assert ft_name, 'has finetune target but no target (ft) name was given'
        logging.info("Mapped model.")
        ft_output_path = os.path.join(out_path, ft_name+'_mapped_'+str(data_seed))
    logging.info("ft out:", ft_output_path)
    model_config = load_config(args.model_config_path)
    pt_config = load_config(args.pretrain_config_path)
    if not args.pretrain_only:
        ft_config = load_config(args.finetune_config_path)
        logging.info("ft config loaded.")
    else:
        ft_config = None
        logging.info("No ft.")
    pretrain_and_finetune(pretrain_outpath, ft_output_path, mapping_model, model_config, pt_config, ft_config, args.eval_only,args.pretrain_only, args.truncate_at, args.load_checkpoint, data_seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out', type=str, required=True)
    parser.add_argument('-p', '--pretrain_name', type=str, required=True)
    parser.add_argument('-f', '--ft_name', type=str, required=False)
    parser.add_argument('-c','--is_common_pretrain',type=bool, default=True)
    parser.add_argument('-m','--mapping_model',type=bool, default=False)
    parser.add_argument('-P','--pretrain_only',type=bool, default=False)
    parser.add_argument('-E','--eval_only',type=bool, default=False)
    parser.add_argument('--model_config_path',type=str, required=True)
    parser.add_argument('--pretrain_config_path',type=str, required=True)
    parser.add_argument('--finetune_config_path',type=str, required=False)
    parser.add_argument('--truncate_at',type=int, required=False, default=-1)
    parser.add_argument('--load_checkpoint',type=bool, required=False, default=True)
    parser.add_argument('--data_seed',type=int, required=False, default=10)
    args = parser.parse_args()
    logging.info(vars(args))
    train(args)
