from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
import os
import logging

#
#
#
#
#


def save_data(args, path, train_set, dev_set, test_set):

    data = train_set.filter(['text', 'label'])
    data.to_csv(os.path.join(path, 'train.tsv'), sep='\t', index=False)

    data = dev_set.filter(['text', 'label'])
    data.to_csv(os.path.join(path, 'dev.tsv'), sep='\t', index=False)

    data = test_set.filter(['text', 'label'])
    data.to_csv(os.path.join(path, 'test.tsv'), sep='\t', index=False)

    logging.info("Saved {train} training, {dev} dev, {test} test samples to {path}".format(
                 train=len(train_set), dev=len(dev_set), test=len(test_set), path=str(path)))

#
#
#
#
#


def divide_users(args, df_raw):

    # Divide users into train, dev and test sets
    logging.info("Dividing users into train, dev and test sets...")

    train_users = []
    dev_users = []
    test_users = []

    for i in range(args.num_labels):
        class_data = df_raw[df_raw['popularity'] == i]
        user_list = class_data['user_id'].unique()
        tr, t = train_test_split(user_list, test_size=0.2, random_state=args.seed)
        tr, d = train_test_split(tr, test_size=0.2, random_state=args.seed)
        train_users.extend(tr)
        dev_users.extend(d)
        test_users.extend(t)

    # Write train, dev and test users to files
    with open(os.path.join(args.data_dir, "train_users.txt"), "w") as user_file:
        for user in train_users:
            user_file.write(str(user) + "\n")
    with open(os.path.join(args.data_dir, "dev_users.txt"), "w") as user_file:
        for user in dev_users:
            user_file.write(str(user) + "\n")
    with open(os.path.join(args.data_dir, "test_users.txt"), "w") as user_file:
        for user in test_users:
            user_file.write(str(user) + "\n")

    return train_users, dev_users, test_users

#
#
#
#
#


def prepare_data(args):
    QA_SEP_TOKEN = '<QA_SEP>'

    logging.info("Preparing train, dev and test sets...")
    # df_raw = pd.read_csv(args.raw_path, delimiter=',')
    df_raw = pd.read_csv(os.path.join(args.data_dir, "raw.csv"))

    # Concatenate T/Q/A according to --mode argument
    if args.mode == "TA":
        df_raw['text'] = df_raw['question_title'] + QA_SEP_TOKEN + df_raw['answer_text']
    elif args.mode == "QA":
        df_raw['text'] = df_raw['question_text'] + QA_SEP_TOKEN + df_raw['answer_text']
    elif args.mode == "TQA":
        df_raw['text'] = df_raw['question_title'] + QA_SEP_TOKEN + \
            df_raw['question_text'] + QA_SEP_TOKEN + df_raw['answer_text']
    elif args.mode is None or args.mode == "A":
        df_raw['text'] = df_raw['answer_text']

    df_raw['label'] = df_raw['popularity']

    # Get train, dev and test users
    train_users, dev_users, test_users = divide_users(args, df_raw)

    # Divide examples into train, dev and test sets according to users
    train = df_raw[df_raw['user_id'].isin(train_users)]
    dev = df_raw[df_raw['user_id'].isin(dev_users)]
    test = df_raw[df_raw['user_id'].isin(test_users)]

    out_dir = os.path.join(args.data_dir, args.mode)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    save_data(args, out_dir, train, dev, test)

#
#
#
#
#


def read_files(args):
    QA_SEP_TOKEN = '<QA_SEP>'

    # start reading
    read_dir = os.path.join(args.data_dir, args.mode)
    logging.info("Reading train, dev and test sets from {}".format(read_dir))
    train_path = os.path.join(read_dir, 'train.tsv')
    dev_path = os.path.join(read_dir, 'dev.tsv')
    test_path = os.path.join(read_dir, 'test.tsv')

    df_train = pd.read_csv(train_path, delimiter='\t')
    df_dev = pd.read_csv(dev_path, delimiter='\t')
    df_test = pd.read_csv(test_path, delimiter='\t')

    train_articles = df_train.text.values
    train_labels = df_train.label.values
    dev_articles = df_dev.text.values
    dev_labels = df_dev.label.values
    test_articles = df_test.text.values
    test_labels = df_test.label.values

    # tokenize the text with bert ids
    logging.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, max_length=args.MAX_LEN)

    train_ids = []
    train_att_mask = []
    dev_ids = []
    dev_att_mask = []
    test_ids = []
    test_att_mask = []

    logging.info("Tokenizing train set which has {} answers...".format(len(train_articles)))
    for article in train_articles:
        q, a = article.split(QA_SEP_TOKEN)
        encoded_article = tokenizer.encode_plus(q, a, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True, return_attention_mask=True,
                                                return_tensors='pt')
        train_ids.append(encoded_article['input_ids'])
        train_att_mask.append(encoded_article['attention_mask'])

    logging.info("Tokenizing dev set which has {} answers...".format(len(dev_articles)))
    for article in dev_articles:
        q, a = article.split(QA_SEP_TOKEN)
        encoded_article = tokenizer.encode_plus(q, a, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True, return_attention_mask=True,
                                                return_tensors='pt')
        dev_ids.append(encoded_article['input_ids'])
        dev_att_mask.append(encoded_article['attention_mask'])

    logging.info("Tokenizing test set which has {} answers...".format(len(test_articles)))
    for article in test_articles:
        q, a = article.split(QA_SEP_TOKEN)
        encoded_article = tokenizer.encode_plus(q, a, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True, return_attention_mask=True,
                                                return_tensors='pt')
        test_ids.append(encoded_article['input_ids'])
        test_att_mask.append(encoded_article['attention_mask'])

    logging.info("Converting train, dev and test sets to torch tensors...")
    train_ids = torch.cat(train_ids, dim=0)
    dev_ids = torch.cat(dev_ids, dim=0)
    test_ids = torch.cat(test_ids, dim=0)
    train_att_mask = torch.cat(train_att_mask, dim=0)
    dev_att_mask = torch.cat(dev_att_mask, dim=0)
    test_att_mask = torch.cat(test_att_mask, dim=0)
    train_labels = torch.tensor(train_labels)
    dev_labels = torch.tensor(dev_labels)
    test_labels = torch.tensor(test_labels)

    train_dataset = TensorDataset(train_ids, train_att_mask, train_labels)
    dev_dataset = TensorDataset(dev_ids, dev_att_mask, dev_labels)
    test_dataset = TensorDataset(test_ids, test_att_mask, test_labels)

    return train_dataset, dev_dataset, test_dataset
