import torch
import logging
import os
import argparse
import sys
from transformers import set_seed
from tokenizers import BertWordPieceTokenizer

set_seed(20)
logging.info(torch.cuda.is_available())

def main(args):
    special_tokens = args.special_tokens
    vocab_size = args.vocab_size
    """We recommend training a byte-level BPE (rather than let’s say,
    a WordPiece tokenizer like BERT) because it will start building
    its vocabulary from an alphabet of single bytes, so all words will
    be decomposable into tokens (no more <unk> tokens!)."""
    # Initialize a tokenizer
    data_paths = args.data_list
    out_path = args.output_path
    tokenizer = BertWordPieceTokenizer()
    # Customize training
    logging.info(f"Training tokenizer on:\n{data_paths}")
    tokenizer.train(
                    files=data_paths,
                    vocab_size=vocab_size,
                    min_frequency=3,
                    show_progress=True,
                    limit_alphabet=5000,
                    )
    logging.info(f"Saving tokenizer at {out_path}")
    os.makedirs(out_path, exist_ok=True)
    tokenizer.save_model(out_path)
    with open(sys.argv[0], 'r') as cur_file:
        cur_running = cur_file.readlines()
    with open(os.path.join(out_path,'script.py'),'w') as log_file:
        log_file.writelines(cur_running)
    with open(os.path.join(out_path,'args.txt'),'w') as log_file:
        log_file.writelines(sys.argv[1:])
    logging.info("Done creating tokenizer")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--special_tokens', default=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], nargs='+', help='list of special tokens for the tokenizer')
    parser.add_argument('-l','--data_list', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-o', '--output_path', type=str, required=True)
    parser.add_argument('-v', '--vocab_size', type=int, required=True)
    args = parser.parse_args()
    main(args)
