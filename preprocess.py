from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import pandas as pd
import os
import logging







def save_data(args, path, train_set, dev_set, test_set):

    data = train_set.filter(['text', 'label'])
    data.to_csv(os.path.join(path, 'train.tsv'), sep='\t', index=False)

    data = dev_set.filter(['text', 'label'])
    data.to_csv(os.path.join(path, 'dev.tsv'), sep='\t', index=False) 
       
    data = test_set.filter(['text', 'label'])
    data.to_csv(os.path.join(path, 'test.tsv'), sep='\t', index=False)

    logging.info("Saved " +str(len(train_set))+ " training, " +str(len(dev_set))+ " dev, " +str(len(test_set))+ " test examples\nTo: " +str(path))



#
#
#
#
#

def prepare_data(args):
    QA_SEP_TOKEN = '<QA_SEP>'

    logging.info("Preparing train/dev/test sets...")
    path = args.raw_path
    df_raw = pd.read_csv(path, delimiter=',')


    #divide the users into train dev and test
    users = {}
    train_users = []
    dev_users = []
    test_users = []
    for i in range(args.num_labels):
        class_data = df_raw[df_raw['popularity']==i]
        user_list = class_data['user_id'].unique()
        tr, t = train_test_split(user_list, test_size=0.2, random_state=args.seed)
        tr, d = train_test_split(tr, test_size=0.2, random_state=args.seed)
        train_users.extend(tr)
        dev_users.extend(d)         
        test_users.extend(t)
    #class_data = df_raw[(df_raw['popularity']==10) | (df_raw['popularity']==11)]
    #user_list = class_data['user_id'].unique()
    #tr, t = train_test_split(user_list, test_size=0.2, random_state=args.seed)
    #tr, d = train_test_split(tr, test_size=0.2, random_state=args.seed)
    #train_users.extend(tr)
    #dev_users.extend(d)         
    #test_users.extend(t)


    #write the selected users to a file
    with open(os.path.join(args.data_dir, "train_users.txt"),"w") as user_file:
        for user in train_users:
            user_file.write(str(user) + "\n")
    with open(os.path.join(args.data_dir, "dev_users.txt"),"w") as user_file:
        for user in dev_users:
            user_file.write(str(user) + "\n")
    with open(os.path.join(args.data_dir, "test_users.txt"),"w") as user_file:
        for user in test_users:
            user_file.write(str(user) + "\n")


    #divide examples into train dev and test according to the user
    train = df_raw[df_raw['user_id'].isin(train_users)]
    dev = df_raw[df_raw['user_id'].isin(dev_users)]
    test = df_raw[df_raw['user_id'].isin(test_users)]


    #construct the datasets
    #for TA
    train_TA = train.copy()
    dev_TA = dev.copy()
    test_TA = test.copy()
    path = os.path.join(args.data_dir, "TA")

    train_TA['text'] = train_TA['question_title'] + QA_SEP_TOKEN + train_TA['answer_text']
    train_TA['label'] = train_TA['popularity']
    dev_TA['text'] = dev_TA['question_title'] + QA_SEP_TOKEN + dev_TA['answer_text']
    dev_TA['label'] = dev_TA['popularity']
    test_TA['text'] = test_TA['question_title'] + QA_SEP_TOKEN + test_TA['answer_text']
    test_TA['label'] = test_TA['popularity']

    save_data(args, path, train_TA, dev_TA, test_TA)

    #for TQA
    train_TQA = train.copy()
    dev_TQA = dev.copy()
    test_TQA = test.copy()
    path = os.path.join(args.data_dir, "TQA")

    train_TQA['text'] = train_TQA['question_title'] + QA_SEP_TOKEN + train_TQA['question_text'] + QA_SEP_TOKEN + train_TQA['answer_text']
    train_TQA['label'] = train_TQA['popularity']
    dev_TQA['text'] = dev_TQA['question_title'] + QA_SEP_TOKEN + dev_TQA['question_text'] + QA_SEP_TOKEN + dev_TQA['answer_text']
    dev_TQA['label'] = dev_TQA['popularity']
    test_TQA['text'] = test_TQA['question_title'] + QA_SEP_TOKEN + test_TQA['question_text'] + QA_SEP_TOKEN + test_TQA['answer_text']
    test_TQA['label'] = test_TQA['popularity']

    save_data(args, path, train_TQA, dev_TQA, test_TQA)


#
#
#
#
#

def read_files(args):
    QA_SEP_TOKEN = '<QA_SEP>'


    # start reading
    logging.info("Reading train/dev/test sets...")
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
    logging.info("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, max_length=args.MAX_LEN)

    train_ids = []
    train_att_mask = []
    dev_ids = []
    dev_att_mask = []
    test_ids = []
    test_att_mask = []

    logging.info("Tokenizing train set...")
    for article in train_articles:
        q, a = article.split(QA_SEP_TOKEN)
        encoded_article = tokenizer.encode_plus(q, a, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        train_ids.append(encoded_article['input_ids'])
        train_att_mask.append(encoded_article['attention_mask'])

    logging.info("Tokenizing dev set...")
    for article in dev_articles:
        q, a = article.split(QA_SEP_TOKEN)
        encoded_article = tokenizer.encode_plus(q, a, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        dev_ids.append(encoded_article['input_ids'])
        dev_att_mask.append(encoded_article['attention_mask'])

    logging.info("Tokenizing test set...")
    for article in test_articles:
        q, a = article.split(QA_SEP_TOKEN)
        encoded_article = tokenizer.encode_plus(q, a, add_special_tokens=True, max_length=args.MAX_LEN,
                                                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
        test_ids.append(encoded_article['input_ids'])
        test_att_mask.append(encoded_article['attention_mask'])

    logging.info("Converting train/dev/test sets to torch tensors...")
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
