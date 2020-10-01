import sys
import logging
import argparse
import random
import coloredlogs
import torch
import numpy as np
import pandas as pd
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import TensorDataset
from torch.utils.data import SequentialSampler, DataLoader

from preprocess import read_files, prepare_data, tokenize_data, read_tokenized_data

# Setup colorful logging
logging.basicConfig()
logger = logging.getLogger('predict.py')
coloredlogs.install(level='WARNING', logger=logger)


def init_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def load_model(checkpoint):
    if checkpoint['model'] == 'bert':
        logger.warning('Preparing BERT classifier...')
        config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=checkpoint['num_labels'])
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
    elif checkpoint['model'] == 'distilbert':
        logger.warning('Preparing DistilBERT classifier...')
        config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=checkpoint['num_labels'])
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

    if checkpoint['device'] == 'cuda':
        if torch.cuda.is_available():
            logger.warning('Running on GPU.')
            model.cuda()
        else:
            logger.error("Checkpoint device ('cuda') is not available!")
            sys.exit()
    else:
        logger.warning('Running on CPU.')
        model.cpu()

    optimizer = AdamW(model.parameters(), lr=checkpoint['lr'], eps=1e-8)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def tokenize_helper(questions, answers, tokenizer, seq, max_len):
    ids = []
    att_mask = []
    if seq == 'A':
        for answer in answers:
            encoded_text = tokenizer.encode_plus(answer, add_special_tokens=True, max_length=max_len,
                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
            ids.append(encoded_text['input_ids'])
            att_mask.append(encoded_text['attention_mask'])
    else:
        for (question, answer) in zip(questions, answers):
            encoded_text = tokenizer.encode_plus(question, answer, add_special_tokens=True, max_length=max_len,
                pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
            ids.append(encoded_text['input_ids'])
            att_mask.append(encoded_text['attention_mask'])
    return ids, att_mask


def tokenize(df, checkpoint):
    questions = df.question.values
    answers = df.answer.values
    labels = df.label.values

    logger.warning("QUESTIONS:\n {}".format(questions))
    logger.warning("ANSWERS:\n {}".format(answers))
    logger.warning("LABELS:\n {}".format(labels))

    if checkpoint['model'] == 'bert':
        logger.warning("Loading BERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    elif checkpoint['model'] == 'distilbert':
        logger.warning("Loading DistilBERT tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    logger.warning("Tokenizing input...")
    ids, att_mask = tokenize_helper(questions, answers, tokenizer, checkpoint['sequence'], checkpoint['max_len'])
    logger.warning("Converting tokenized input to torch tensor...")
    ids = torch.cat(ids,dim=0)
    att_mask = torch.cat(att_mask,dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(ids, att_mask, labels)

    return dataset


def test(test_iter, model, device):
    model.eval()
    preds = []
    trues = []
    for batch_ids in test_iter:
        input_ids = batch_ids[0].to(device)
        att_masks = batch_ids[1].to(device)
        labels = batch_ids[2].to(device)

        with torch.no_grad():
            _, logits = model(input_ids, attention_mask=att_masks, labels=labels)

        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)

    return trues, preds


def get_df_inputs():
    question = input('Enter question (as plain text): ')
    answer = input('Enter answer (as plain text): ')
    label = int(input('Enter label (as 0 or 1): '))
    data_dict = {'question': [question], 'answer': [answer], 'label': [label]}
    data_frame = pd.DataFrame(data=data_dict)
    return data_frame

def main():
    parser = argparse.ArgumentParser(description='Grey Literature Predict')
    parser.add_argument('--checkpoint_path', default=None, type=str, required=True)
    parser.add_argument('--test_path', default=None, type=str)

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint_path)

    logger.warning(("LOADED CHECKPOINT FROM {} \n{{"
                  "\n  epoch: {}\n  dev_score: {:.4f}\n  test_score: {:.4f}\n  seed: {}\n  lr: {}"
                  "\n  device: {}\n  model: {}\n  labels: {}"
                  "\n  num_labels: {}\n  sequence: {}\n  crop: {}\n  max_len: {}\n}}").format(
                      args.checkpoint_path, checkpoint['epoch'], checkpoint['dev_score'],
                      checkpoint['test_score'], checkpoint['seed'], checkpoint['lr'],
                      checkpoint['device'], checkpoint['model'], checkpoint['labels'],
                      checkpoint['num_labels'], checkpoint['sequence'], checkpoint['crop'],
                      checkpoint['max_len']
                  ))

    init_random_seeds(checkpoint['seed'])

    if args.test_path == None:
        df_test = get_df_inputs()
    else:
        df_test = pd.read_csv(args.test_path, delimiter=',')

    test_dataset = tokenize(df_test, checkpoint)

    test_iter = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=1)

    model, _ = load_model(checkpoint)
    test_label, test_pred = test(test_iter, model, checkpoint['device'])

    pred_class = np.concatenate([np.argmax(numarray, axis=1) for numarray in test_pred]).ravel()
    label_class = np.concatenate([numarray for numarray in test_label]).ravel()

    print('Expected:  {}'.format(label_class))
    print('Predicted: {}'.format(pred_class))


if __name__ == '__main__':
    main()
