import argparse
import json
import os


def create_pretrain_config(train_data_paths_list, eval_data_paths_list, num_epochs, batch_size, name):
    pretrain_config = {
        'train_data_paths_list':train_data_paths_list,
        'eval_data_paths_list':eval_data_paths_list,
        'num_epochs':num_epochs,
        'batch_size':batch_size
    }
    with open(os.path.join(OUT_PATH,f'{name}.json'), 'w') as fp:
        json.dump(pretrain_config, fp)


def create_ft_config(train_data_paths_list, eval_data_paths_list, num_epochs, batch_size, name):
    ft_config = {
        'train_data_paths_list':train_data_paths_list,
        'eval_data_paths_list':eval_data_paths_list,
        'num_epochs':num_epochs,
        'batch_size':batch_size
    }
    with open(os.path.join(OUT_PATH,f'{name}.json'), 'w') as fp:
        json.dump(ft_config, fp)


def create_model_config(tokenizer_path, hidden_layer_size, vocab_size, max_sent_len,num_hidden,num_attention, name):
    model_config = {
        'tokenizer_path':tokenizer_path,
        'hidden_layer_size':hidden_layer_size,
        'vocab_size':vocab_size,
        'max_sent_len':max_sent_len,
        'num_hidden':num_hidden,
        'num_attention':num_attention
    }
    with open(os.path.join(OUT_PATH,f'{name}.json'), 'w') as fp:
        json.dump(model_config, fp)


def create_configs(args):
    factory = {'pretrain':{'func':create_pretrain_config,'args':[args.pt_train_data_list, args.pt_eval_data_list, args.pt_num_epochs, args.pt_batch_size, args.pt_name]},
               'finetune':{'func':create_ft_config, 'args':[args.ft_train_data_list, args.ft_eval_data_list, args.ft_num_epochs, args.ft_batch_size, args.ft_name]},
               'model':{'func':create_model_config, 'args':[args.tokenizer_path, args.hidden_size, args.vocab_size, args.max_len, args.num_hidden, args.num_attention, args.model_name]}
               }
    config_types = args.config_types
    for type in config_types:
        assert type in factory
        factory[type]['func'](*factory[type]['args'])


def check_input(config_types):
    if 'pretrain' in config_types:
        assert args.pt_train_data_list and args.pt_eval_data_list and args.pt_num_epochs and args.pt_batch_size and args.pt_name
    if 'finetune' in config_types:
        assert args.ft_train_data_list and args.ft_eval_data_list and args.ft_num_epochs and args.ft_batch_size and args.ft_name
    if 'model' in config_types:
        assert args.tokenizer_path and args.hidden_size and args.vocab_size and args.max_len and args.num_hidden and args.num_attention and args.model_name


def main(args):
    check_input(args.config_types)
    create_configs(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c','--config_types', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('-o','--out_path', type=str, help='<Required> Set flag', required=True)

    parser.add_argument('--pt_train_data_list', nargs='+', help='<Required> Set flag', default=None)
    parser.add_argument('--pt_eval_data_list', nargs='+', help='<Required> Set flag', default=None)
    parser.add_argument('--pt_num_epochs',type=int, default=None)
    parser.add_argument('--pt_batch_size',type=int, default=None)
    parser.add_argument('--pt_name',type=str, default=None)

    parser.add_argument('--ft_train_data_list', nargs='+', help='<Required> Set flag', default=None)
    parser.add_argument('--ft_eval_data_list', nargs='+', help='<Required> Set flag', default=None)
    parser.add_argument('--ft_num_epochs',type=int, default=None)
    parser.add_argument('--ft_batch_size',type=int, default=None)
    parser.add_argument('--ft_name',type=str, default=None)

    parser.add_argument('--tokenizer_path',type=str, default=None)
    parser.add_argument('--vocab_size',type=int, default=None)
    parser.add_argument('--hidden_size',type=int, default=None)
    parser.add_argument('--max_len',type=int, default=None)
    parser.add_argument('--num_hidden',type=int, default=None)
    parser.add_argument('--num_attention',type=int, default=None)
    parser.add_argument('--model_name',type=str, default=None)

    args = parser.parse_args()
    OUT_PATH = args.out_path
    os.makedirs(OUT_PATH, exist_ok=True)
    assert OUT_PATH
    main(args)
