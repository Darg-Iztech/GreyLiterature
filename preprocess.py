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


def read_files(args, method='combined', prepare=False):

    # prepare train dev and test files, if set
    if prepare:
        path = args.raw_path
        df_raw = pd.read_csv(path, delimiter=',')
        # q = df_raw.question_text.values
        # a = df_raw.answer_text.values
        # label = df_raw.popularity.values

        if method == 'combined':
            df_raw['text'] = df_raw['question_text'] + df_raw['answer_text']
        else:
            df_raw['text'] = df_raw['answer_text']

        df_raw['label'] = df_raw['popularity']
        data = df_raw.filter(['text', 'label'])

        train, test = train_test_split(data, test_size=0.2, random_state=args.seed)
        train, dev = train_test_split(train, test_size=0.2, random_state=args.seed)
        train.to_csv(os.path.join(args.data_dir, 'train.tsv'), sep='\t', index=False)
        dev.to_csv(os.path.join(args.data_dir, 'dev.tsv'), sep='\t', index=False)
        test.to_csv(os.path.join(args.data_dir, 'test.tsv'), sep='\t', index=False)

    # start reading
    logging.debug("Reading Datasets...")
    train_path = os.path.join(args.data_dir, 'train.tsv')
    dev_path = os.path.join(args.data_dir, 'dev.tsv')
    test_path = os.path.join(args.data_dir, 'test.tsv')
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
    logging.debug("Tokenizing the Dataset for BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    train_ids = []
    train_att_mask = []
    dev_ids = []
    dev_att_mask = []
    test_ids = []
    test_att_mask = []

    for article in train_articles:
        encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        train_ids.append(encoded_article['input_ids'])
        train_att_mask.append(encoded_article['attention_mask'])
    for article in dev_articles:
        encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        dev_ids.append(encoded_article['input_ids'])
        dev_att_mask.append(encoded_article['attention_mask'])
    for article in test_articles:
        encoded_article = tokenizer.encode_plus(article, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        test_ids.append(encoded_article['input_ids'])
        test_att_mask.append(encoded_article['attention_mask'])

    # convert the dataset to torch tensors
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
