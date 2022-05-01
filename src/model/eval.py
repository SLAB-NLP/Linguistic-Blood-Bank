import torch
from tqdm import tqdm
from transformers import AutoModelForMaskedLM
from transformers import BertTokenizer, set_seed,BertForMaskedLM, BertConfig
import os
import json
import numpy as np
import argparse
from dataset import LineByLineTextDataset, DataCollatorForLanguageModeling
import logging
logging.info(torch.cuda.is_available())
set_seed(10)

def eval_single_model(args):
    """
    runs a single evaluation that produces an average MRR score over the sentences in the input files. The MRR score is
    saved in the specified path.
    :param args: reffer to the 'help' arguments in the argparse section below.
    """
    # initializing variables w.r.t args:
    model_dir_path = args.model_dir_path
    model_config_path = args.model_config_path
    eval_data_paths = args.eval_data_paths
    truncate_at = args.truncate_at
    rand_model = args.rand_model
    overrite = args.overrite
    is_zero_shot = args.is_zero_shot
    out_path = args.out_path if args.out_path is not None else model_dir_path
    base_name = 'mrr_eval_' if not is_zero_shot else 'mrr_eval_zero_shot'
    logging.info("Output is: ", out_path, ", with base name ", base_name)
    config =json.load(open(model_config_path,'r'))
    tokenizer = BertTokenizer.from_pretrained(config['tokenizer_path'])
    if truncate_at >= 1:
        if os.path.exists(os.path.join(out_path,f'{base_name+str(truncate_at)}.txt')) and not overrite:
            logging.info(f"stats already exist at {os.path.join(out_path,f'{base_name+str(truncate_at)}.txt')}, no overrite. returning. ")
            return
    if truncate_at < 1:
        if os.path.exists(os.path.join(out_path,f'{base_name}all.txt')) and not overrite:
            logging.info(f"stats already exist at {os.path.join(out_path,f'{base_name+str(truncate_at)}.txt')}, no overrite. returning.")
            return
    logging.info("Gathering stats...")
    if rand_model:
        hg_config = BertConfig(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_layer_size'],
            num_hidden_layers=config['num_hidden'],
            num_attention_heads=config['num_attention'],
            max_position_embeddings=config['max_sent_len']
        )
        model = BertForMaskedLM(hg_config)
    else:
        model = AutoModelForMaskedLM.from_pretrained(model_dir_path)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    if truncate_at >=1:
        ft_eval = LineByLineTextDataset(tokenizer=tokenizer, file_paths= eval_data_paths, block_size=config['max_sent_len'], truncate_at=truncate_at, randomize=False)
    else:
        ft_eval = LineByLineTextDataset(tokenizer=tokenizer, file_paths= eval_data_paths, block_size=config['max_sent_len'], randomize=False)
    logging.info(f"Evaulating {model_dir_path} on {eval_data_paths} with truncate {truncate_at} and zeroshot {is_zero_shot}. Overrite:{overrite}.")
    # gathering scores:
    batch_size = 128
    mrrs = []
    rank_accs = []
    num_of_masked = []
    for idx in tqdm(range(0,len(ft_eval), batch_size), desc='getting eval results...'):
        data_dict = data_collator([ft_eval[i]['input_ids'] for i in range(min(batch_size, len(ft_eval)-idx))]) #dict: {'input_ids':<tokens>, 'labels':<-100 should be ignored>}
        inputs, labels = data_collator.mask_tokens(data_dict['input_ids'])
        logits = model(inputs).logits
        masked_indices = labels != -100
        masked_words_num = np.count_nonzero(masked_indices)
        preds_logits = logits[masked_indices].detach().numpy()
        num_of_masked.append(masked_words_num)
        top_preds_indices = np.argsort(-preds_logits)
        lables_at_mask = labels[masked_indices].numpy().reshape(-1,1)
        correct_word_indices = np.argwhere(lables_at_mask == top_preds_indices)
        if sum(correct_word_indices.shape) == 0:
            mrrs.append(0)
            logging.info("No cands found.")
            continue
        else:
            correct_word_indices = correct_word_indices[:, -1]
        mrrs.append((np.sum(1/(correct_word_indices+1))/masked_words_num))
        rank_accs.append(np.sum(1-(correct_word_indices/config['vocab_size']))/len(correct_word_indices))
    logging.info(f"BERT mrr: {sum(mrrs)/len(mrrs)}")
    logging.info(f"BERT rank accuracy: {sum(rank_accs)/len(rank_accs)}")
    # saving the stats:
    if truncate_at < 1:
        truncate_at = "all"
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path,f'{base_name+str(truncate_at)}.txt'), 'w') as eval_mrr_out:
        json.dump({'eval_mrr':sum(mrrs)/len(mrrs), 'all_batches_mrrs':mrrs, 'num_of_masked_per_batch':num_of_masked}, eval_mrr_out)
    with open(os.path.join(out_path,f'rank_acc_eval_{base_name+str(truncate_at)}.txt'), 'w') as eval_rank_acc_out:
        json.dump({'eval_rank_acc':sum(rank_accs)/len(mrrs), 'all_batches_rank_acc':rank_accs}, eval_rank_acc_out)
    with open(os.path.join(out_path,f'script_args_{base_name+str(truncate_at)}.txt'), 'w') as script_args:
        json.dump({'args':str(vars(args)),'note':'overrite is manually always True.'}, script_args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval_data_paths', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-m', '--model_dir_path', type=str, help='<Required> Set flag', required=True)
    parser.add_argument('-c', '--model_config_path',type=str, required=True)
    parser.add_argument('-t', '--truncate_at',type=int, default=-1)
    parser.add_argument('--overrite',type=bool, default=True)
    parser.add_argument('-z', '--is_zero_shot',type=bool, default=False)
    parser.add_argument('-o', '--out_path',type=str, default=None)
    parser.add_argument('-r', '--rand_model',type=bool, default=False)
    args = parser.parse_args()
    logging.info(vars(args))
    eval_single_model(args)