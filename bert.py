from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import numpy as np
# import pandas as pd
import os
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

stats_columns = '{0:>5}|{1:>5}|{2:>5}|{3:>5}|{4:>5}|{5:>5}|{6:>5}|{7:>5}|{8:>5}|{9:>5}|{10:>5}'

#
#
#
#
#


def run(model, train_data, dev_data, test_data, optimizer, args):

    train_iter = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=args.batch_size)
    dev_iter = DataLoader(dev_data, sampler=SequentialSampler(dev_data), batch_size=args.batch_size)
    test_iter = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=args.batch_size)

    torch.cuda.empty_cache()

    logging.info("Number of training samples {train}, number of dev samples {dev}, number of test samples {test}"
                 .format(train=len(train_data), dev=len(dev_data), test=len(test_data)))

    train(train_iter, dev_iter, model, optimizer, args)

    _test_label, _test_pred, test_loss = test(test_iter, model, args)

    # TO DO check if the metrics hold for the multi-class classification
    test_acc, test_f1, test_recall, test_prec = calculate_metrics(_test_label, _test_pred)
    logging.info("TEST RESULTS:\nAccuracy: {acc}\nF1: {F1}\nRecall: {recall}\nPrecision: {prec}".format(
                 acc=test_acc, f1=test_f1, recall=test_recall, prec=test_prec))
#
#
#
#
#


def train(train_iter, dev_iter, model, optimizer, args):
    best_dev_f1 = -1

    n_total_steps = len(train_iter)
    total_iter = len(train_iter) * args.epochs

    logging.info(
        stats_columns.format(
            'Epoch', 'T-Acc', 'T-F1', 'T-Recall', 'T-Prec', 'T-Loss',
            'D-Acc', 'D-F1', 'D-Recall', 'D-Prec', 'D-Loss'))

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_iter)

    for epoch in range(args.epochs):

        model.train()

        train_loss = 0
        preds = []
        trues = []

        for batch_ids in train_iter:

            input_ids = batch_ids[0].to(args.device)
            att_masks = batch_ids[1].to(args.device)
            labels = batch_ids[2].to(args.device)

            model.zero_grad()

            # forward pass
            loss, logits = model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)

            # record preds, trues
            _pred = logits.cpu().data.numpy()
            preds.append(_pred)
            _label = labels.cpu().data.numpy()
            trues.append(_label)

            train_loss += loss.item()

            # backpropagate and update optimizer learning rate
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        train_loss = train_loss / n_total_steps

        train_acc, train_f1, train_recall, train_prec = calculate_metrics(trues, preds)

        _dev_label, _dev_pred, dev_loss = eval(dev_iter, model, args)

        dev_acc, dev_f1, dev_recall, dev_prec = calculate_metrics(_dev_label, _dev_pred)

        logging.info(
            stats_columns.format(epoch, train_acc, train_f1, train_recall, train_prec, train_loss,
                                 dev_acc, dev_f1, dev_recall, dev_prec, dev_loss))

        if best_dev_f1 < dev_f1:
            logging.info('New dev acc {dev_acc} is larger than best dev acc {best_dev_acc}'.format(
                         dev_acc=dev_f1, best_dev_acc=best_dev_f1))
            best_dev_f1 = dev_f1
            model_name = 'epoch_{epoch}_dev_f1_{dev_f1:03}.pth.tar'.format(epoch=epoch, dev_f1=dev_f1)
            save_model(model, optimizer, epoch, model_name, args.checkpoint_dir)

#
#
#
#
#


def eval(dev_iter, model, args):
    n_total_steps = len(dev_iter)
    model.eval()
    dev_loss = 0
    preds = []
    trues = []
    for batch_ids in dev_iter:
        input_ids = batch_ids[0].to(args.device)
        att_masks = batch_ids[1].to(args.device)
        labels = batch_ids[2].to(args.device)

        # forward pass
        with torch.no_grad():
            loss, logits = model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)
        dev_loss += loss.item()

        # record preds, trues
        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)

    dev_loss = dev_loss / n_total_steps
    return trues, preds, dev_loss


#
#
#
#
#

def test(test_iter, model, args):
    n_total_steps = len(test_iter)
    model.eval()
    test_loss = 0
    preds = []
    trues = []
    for batch_ids in test_iter:
        input_ids = batch_ids[0].to(args.device)
        att_masks = batch_ids[1].to(args.device)
        labels = batch_ids[2].to(args.device)

        # forward pass
        with torch.no_grad():
            loss, logits = model(input_ids, token_type_ids=None, attention_mask=att_masks, labels=labels)
        test_loss += loss.item()

        # record preds, trues
        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)

    test_loss = test_loss / n_total_steps
    return trues, preds, test_loss


#
#
#
#
#


def calculate_metrics(label, pred):
    pred_class = np.concatenate([np.argmax(numarray, axis=1) for numarray in pred]).ravel()
    label_class = np.concatenate([numarray for numarray in label]).ravel()

    logging.info('Expected: \n{}'.format(label_class[:20]))
    logging.info('Predicted: \n{}'.format(pred_class[:20]))
    acc = accuracy_score(label_class, pred_class)
    f1 = f1_score(label_class, pred_class, average='binary')
    recall = recall_score(label_class, pred_class)
    prec = precision_score(label_class, pred_class)

    return acc, f1, recall, prec

#
#
#
#
#


def save_model(model, optimizer, epoch, model_name, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_path = os.path.join(checkpoint_dir, model_name)

    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }, save_path)

    logging.info('Best model is saved to {save_path}'.format(save_path=save_path))

    return save_path

#
#
#
#
#


def load_model(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    logging.info('Loaded checkpoint from path "{}" (at epoch {})'.format(
                 checkpoint_path, checkpoint['epoch']))
