import logging
import argparse
import coloredlogs
import torch
import os
from torch import nn
from torch import optim
from transformers import BertForSequenceClassification, BertConfig, AdamW

from preprocess import read_files
from bert import run

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('main_bert.py')
logger.root.setLevel(logging.DEBUG)
coloredlogs.install(level='DEBUG', logger=logger)

random_seed = 42


def init_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Grey Literature Tests')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--checkpoint_dir', default='./models')
    parser.add_argument('--MAX_LEN', type=int, default=512)


    args = parser.parse_args()
    #args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = 'cpu'
    init_random_seeds(args.seed)



    logging.debug('Reading datasets...')
    args.raw_path = '/media/darg1/Data/Projects/Grey Litrerature/data/CSV/design_patterns_users/one_question_for_each_user'
    args.data_path = '/media/darg1/Data/Projects/Grey Litrerature/GreyLiterature/data'
    train_data, dev_data, test_data = read_files(args, method='combined', prepare=False)


    logging.debug('Starting Experiments...')    
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2,
                                                          output_attentions=False, output_hidden_states=False)
    model.cpu()
    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8)
    run(model, train_data, dev_data, optimizer, args)








    #todo evaluate with test data
