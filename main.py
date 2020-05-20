import logging
import argparse
import coloredlogs
import torch
from str2bool import str2bool
from torch.utils.data import Subset
import os
# import numpy as np
# from torch import nn
# from torch import optim
# from transformers import BertConfig
from transformers import BertForSequenceClassification, AdamW

from preprocess import read_files, prepare_data
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
    parser.add_argument('--device', default='cpu', type=str, help="Use cuda if available")
    parser.add_argument('--checkpoint_dir', default='./models', type=str)
    parser.add_argument('--MAX_LEN', default=512, type=int)

    parser.add_argument('--data_dir', default=None, required=True, type=str)
    parser.add_argument('--prepare', default=False, type=str2bool)
    parser.add_argument('--experiment', default=False, type=str2bool)
    parser.add_argument('--mode', default="TA", type=str, help="Concatenates title, Q and A (options: QA, A, TQA, TA)")
    parser.add_argument('--raw_path', default=None, type=str, help="Required if --prepared=True")
    parser.add_argument('--num_labels', default=12, type=int, required=True, help="Number of classes in dataset")

    args = parser.parse_args()
    # args, unknown = parser.parse_known_args()  # use this verion in jupyter notebooks to avoid conflicts

    if args.prepare and (args.raw_path is None):
        parser.error("--prepare requires --raw_path")

    init_random_seeds(args.seed)

    if args.prepare:
        prepare_data(args)
    train_data, dev_data, test_data = read_files(args)

    # run for a small subset, if set
    if args.experiment:
        logging.info('Running in experiment mode! Subsetting the datasets...')
        train_data = Subset(train_data, range(0, 640))
        dev_data = Subset(dev_data, range(0, 160))
        test_data = Subset(test_data, range(0, 200))



    logging.info('Starting executions...')
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=args.num_labels,
                                                          output_attentions=False, output_hidden_states=False)
    model.cpu()
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)

    run(model, train_data, dev_data, test_data, optimizer, args)

    # todo evaluate with test data
