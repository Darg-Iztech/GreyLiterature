import logging
import argparse
import random
import coloredlogs
import torch
from str2bool import str2bool
# import os
import numpy as np
# from torch import nn
# from torch import optim
# from transformers import BertConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, AdamW

from preprocess import read_files, prepare_data, tokenize_data, read_tokenized_data
from bert import run

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)

default_random_seed = 42


def init_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grey Literature Tests')
    parser.add_argument('--seed', default=default_random_seed, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--device', default='cpu', type=str, help="(options ['cpu', 'cuda'] defaults to 'cpu')")
    parser.add_argument('--checkpoint_dir', default='./models', type=str)
    parser.add_argument('--MAX_LEN', default=512, type=int)
    parser.add_argument('--model', default='bert', type=str, help="(options ['bert', 'distilbert'] defaults to 'bert')")
    parser.add_argument('--data_dir', default=None, required=True, type=str)
    parser.add_argument('--sequence', default='TQA', type=str, help="(options ['A', 'TA', 'QA', 'TQA'] defaults to 'TQA')")
    parser.add_argument('--t_start', default=None, type=str, help=argparse.SUPPRESS)
    parser.add_argument('--labels', default=None, required=True, type=str, help="(options ['sum_class', 'mean_class', 'median_class']")
    parser.add_argument('--num_labels', default=0, type=int)
    parser.add_argument('--crop', default=1.0, type=float,
                        help="If 1 no crop, if 0.25 crop 25%% from top and bottom")
    parser.add_argument('--tokenize', default=True, type=str2bool)  # set false to read pretokenized data
    parser.add_argument('--save_models', default=True, type=str2bool)

    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()  # use this version in jupyter notebooks to avoid conflicts

    init_random_seeds(args.seed)

    # if args.prepare:
    prepare_data(args)

    df_train, df_dev, df_test = read_files(args)

    # automatically identify the number of labels:
    num_labels = len(np.union1d(np.union1d(df_train['label'], df_dev['label']), df_test['label']))
    args.num_labels = num_labels
    logging.info('Identified {} labels in the dataset.'.format(num_labels))

    if args.tokenize:
        train_data, dev_data, test_data = tokenize_data(args, df_train, df_dev, df_test)
    else:
        train_data, dev_data, test_data = read_tokenized_data(args, df_train, df_dev, df_test)

    if args.model == 'bert':
        logging.info('Starting executions with BERT...')
        config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
    elif args.model == 'distilbert':
        logging.info('Starting executions with DistilBERT...')
        config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

    if torch.cuda.is_available() and args.device == 'cuda':
        logging.info('Running on GPU !!!')
        model.cuda()
    else:
        logging.info('Running on CPU !!!')
        model.cpu()

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    run(model, train_data, dev_data, test_data, optimizer, args)
