from collections import Counter
import numpy as np
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

def get_token_info(tokenized_data):
    """
    given a path to tokenized data or a list of tokens, returns the info measures used in the paper: |total_tokens|/|unique_tokens|.
    :param tokenized_data: path or list of tokens.
    :return: total_tokens, unique_tokens, unique_tokens/total_tokens
    """
    if isinstance(tokenized_data, str):
        with open(tokenized_data, 'r') as tokenized_lines_file:
            tokens = [x.strip() for x in tokenized_lines_file.readlines()] # removing \n
    elif isinstance(tokenized_data, list):
        tokens = tokenized_data
    else:
        assert False, 'Inputs must be either a tokens list or a file containing all tokens seperated by \\n'
    token_counts = Counter(tokens)
    total_tokens = np.sum(list(token_counts.values()))
    unique_tokens = len(token_counts)
    logging.info(f"total tokens:{total_tokens}, unique tokens:{unique_tokens}, ratio:{unique_tokens/total_tokens}")
    return total_tokens, unique_tokens, unique_tokens/total_tokens

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tokens_file', type=str, required=True)
    args = parser.parse_args()
    logging.info(vars(args))
    get_token_info(args.tokens_file)
