import logging
import argparse
import coloredlogs
import torch
from str2bool import str2bool
# import os
import numpy as np
# from torch import nn
# from torch import optim
# from transformers import BertConfig
from transformers import AutoConfig, AutoModelForSequenceClassification, AdamW

from preprocess import read_files, prepare_data, tokenize_data, print2logfile
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
    parser.add_argument('--prepare', default=False, type=str2bool, help="(options [True, False] defaults to False)")
    parser.add_argument('--experiment', default=False, type=str2bool, help="(options [True, False] defaults to False)")
    parser.add_argument('--mode', default='TQA', type=str, help="(options ['A', 'TA', 'QA', 'TQA'] defaults to 'TA')")
    parser.add_argument('--num_labels', default=None, type=int, help="Number of classes in dataset")

    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()  # use this verion in jupyter notebooks to avoid conflicts

    init_random_seeds(args.seed)

    if args.prepare:
        prepare_data(args)

    df_train, df_dev, df_test = read_files(args)

    # automatically identify the number of labels:
    args.num_labels = len(np.union1d(np.union1d(df_train['label'], df_dev['label']), df_test['label']))
    logging.info('Identified {} labels in the dataset.'.format(args.num_labels))

    # run for a small subset, if set
    if args.experiment:
        logging.info('Running in experiment mode! Subsetting the datasets...')
        df_train = df_train.head(640)
        df_dev = df_dev.head(160)
        df_test = df_test.head(200)

    train_data, dev_data, test_data = tokenize_data(args, df_train, df_dev, df_test)

    if args.model == 'bert':
        logging.info('Starting executions with BERT...')
        config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=args.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
    elif args.model == 'distilbert':
        logging.info('Starting executions with DistilBERT...')
        config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=args.num_labels)
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        model.cuda()
    else:
        args.device = torch.device('cpu')
        model.cpu()

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    run(model, train_data, dev_data, test_data, optimizer, args)
