from transformers import get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
import numpy as np
import pandas as pd
import datetime as dt
import os
import logging
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report
from preprocess import print2logfile

stats_head = '{0:>5}|{1:>7}|{2:>7}|{3:>7}|{4:>7}|{5:>7}|{6:>7}|{7:>7}|{8:>7}|{9:>7}|{10:>7}|' + \
             '{11:>7}|{12:>7}|{13:>7}|{14:>7}|{15:>7}|{15:>7}|'
stats_values = '{0:>5}|{1:>6.5f}|{2:>6.5f}|{3:>6.5f}|{4:>6.5f}|{5:>6.5f}|{6:>6.5f}|' + \
               '{7:>6.5f}|{8:>6.5f}|{9:>6.5f}|{10:>6.5f}|{11:>6.5f}|{12:>6.5f}|{13:>6.5f}|' + \
               '{14:>6.5f}|{15:>6.5f}|{15:>6.5f}|'
stats_head_vals = ['Epoch', 'Tr-Acc', 'Tr-F1', 'Tr-Rec', 'Tr-Prec', 'Tr-Loss',
                   'Dv-Acc', 'Dv-F1', 'Dv-Rec', 'Dv-Prec', 'Dv-Loss',
                   'Te-Acc', 'Te-F1', 'Te-Rec', 'Te-Prec', 'Te-Loss']
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

    logging.info("Number of training samples {train}, number of dev samples {dev}, number of test samples {test}\n"
                 .format(train=len(train_data), dev=len(dev_data), test=len(test_data)))

    t_start = dt.datetime.now().replace(microsecond=0)
    args.t_start = t_start.strftime("%Y%m%d_%H%M%S")

    train_params = ""
    for param in vars(args):
        train_params += param + '=' + str(getattr(args, param)) + '\n'

    print2logfile("### TRAINING STARTED AT {} WITH PARAMS:\n\n{}".format(t_start, train_params), args)

    train(train_iter, dev_iter, test_iter, model, optimizer, args)

    t_delta = dt.datetime.now().replace(microsecond=0) - t_start
    logging.info("\n\n\n### TRAINING ENDED IN {}".format(t_delta))
    print2logfile("\n\n\n### TRAINING ENDED IN {}".format(t_delta), args)

#
#
#
#
#


def train(train_iter, dev_iter, test_iter, model, optimizer, args):
    best_dev_epoch = 0
    best_test_epoch = 0
    best_dev_epoch_dev_score = -1.0  # acc for binary, f1 for multilabel classification
    best_test_epoch_test_score = -1.0  # acc for binary, f1 for multilabel classification
    prev_best_model_name = ""  # to delete when there is a new best
    classification_type = 'binary' if args.crop < 1.0 else 'multiclass'

    # create df and csv for stats
    stats_df = pd.DataFrame(columns=stats_head_vals)
    dataset_name = args.data_dir.split('/')[-1]  # returns 'dp' or 'se'
    stats_csv_name = '{}_{}_{}_{}_{}_{}.csv'.format(
        args.model, dataset_name, args.sequence, classification_type, args.labels.split('_')[0], args.t_start)
    # example filename: bert_dp_TQA_median_20200609_164520.csv
    stats_csv_path = os.path.join(args.checkpoint_dir, stats_csv_name)
    stats_df.to_csv(stats_csv_path, sep=',', index=False)

    best_stats_head_vals = ['model', 'dataset', 'sequence', 'classification', 'labeling',
                            't_start', 'best_dev_epoch', 'best_dev_epoch_dev_score',
                            'best_dev_epoch_test_score', 'best_test_epoch', 'best_test_epoch_dev_score',
                            'best_test_epoch_test_score', 'seed']
    best_stats_df = pd.DataFrame(columns=best_stats_head_vals)
    best_stats_csv = 'models/best_stats.csv'
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists(best_stats_csv):
        best_stats_df.to_csv(best_stats_csv, sep=',', index=False)  # write heading only if not exists

    n_total_steps = len(train_iter)
    total_iter = len(train_iter) * args.epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_iter)

    for epoch in range(args.epochs):

        print2logfile("\n\n\n\n-------------------------epoch "+ str(epoch) +"-------------------------", args)
        model.train()

        train_loss = 0
        train_step = 1
        preds = []
        trues = []

        for batch_ids in train_iter:

            print(">>>>>>>>> Train step {} / {}".format(train_step, n_total_steps), end="\r")
            input_ids = batch_ids[0].to(torch.device(args.device))
            att_masks = batch_ids[1].to(torch.device(args.device))
            labels = batch_ids[2].to(torch.device(args.device))

            model.zero_grad()

            # forward pass
            loss, logits = model(input_ids, attention_mask=att_masks, labels=labels)

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
            train_step += 1

        print('\n')

        train_loss = train_loss / n_total_steps

        print2logfile("--Training--", args)
        train_acc, train_f1, train_recall, train_prec = calculate_metrics(trues, preds, args)

        _dev_label, _dev_pred, dev_loss = eval(dev_iter, model, args)
        print2logfile("--Validation--", args)
        dev_acc, dev_f1, dev_recall, dev_prec = calculate_metrics(_dev_label, _dev_pred, args)

        _test_label, _test_pred, test_loss = test(test_iter, model, args)
        print2logfile("--Test--", args)
        test_acc, test_f1, test_recall, test_prec = calculate_metrics(_test_label, _test_pred, args)

        print(stats_head.format(*stats_head_vals))
        print(stats_values.format(epoch, train_acc, train_f1, train_recall, train_prec, train_loss,
                                  dev_acc, dev_f1, dev_recall, dev_prec, dev_loss,
                                  test_acc, test_f1, test_recall, test_prec, test_loss))
        print()

        print2logfile(stats_head.format(*stats_head_vals), args)
        print2logfile(stats_values.format(epoch, train_acc, train_f1, train_recall, train_prec, train_loss,
                                dev_acc, dev_f1, dev_recall, dev_prec, dev_loss,
                                test_acc, test_f1, test_recall, test_prec, test_loss), args)

        # append epoch stats to stats csv file
        epoch_stats_df = pd.DataFrame(columns=stats_head_vals)
        epoch_stats_df.at[0] = np.around([epoch, train_acc, train_f1, train_recall, train_prec, train_loss,
                                          dev_acc, dev_f1, dev_recall, dev_prec, dev_loss,
                                          test_acc, test_f1, test_recall, test_prec, test_loss], 4)
        epoch_stats_df.to_csv(stats_csv_path, mode='a', header=False, index=False)




        # CHOOSE METRICS (ACC or F1)
        if args.crop == 1.0:
            dev_score = dev_f1
            dev_score_name = 'f1'
            test_score = test_f1
            # test_score_name = 'f1'
        else:
            dev_score = dev_acc
            dev_score_name = 'acc'
            test_score = test_acc
            # test_score_name = 'acc'

        # NEW BEST TEST SCORE
        if best_test_epoch_test_score < test_score:
            best_test_epoch_dev_score = dev_score
            best_test_epoch_test_score = test_score
            best_test_epoch = epoch

        # NEW BEST DEV SCORE
        if best_dev_epoch_dev_score < dev_score:
            logging.info('New dev {} {:.4f} is larger than the best dev {} {:.4f}'.format(
                         dev_score_name, dev_score, dev_score_name, best_dev_epoch_dev_score))
            print2logfile('\nNew dev {} {:.4f} is larger than the best dev {} {:.4f}'.format(
                         dev_score_name, dev_score, dev_score_name, best_dev_epoch_dev_score), args)

            best_dev_epoch_dev_score = dev_score
            best_dev_epoch_test_score = test_score
            best_dev_epoch = epoch

            dataset = args.data_dir.split('/')[-1]  # returns 'dp' or 'se'
            labeling = args.labels.split('_')[0]

            # update best stats
            best_stats = [args.model, dataset, args.sequence, classification_type,
                labeling, args.t_start, best_dev_epoch, round(best_dev_epoch_dev_score, 4),
                round(best_dev_epoch_test_score, 4)]

            if args.save_models:
                model_name = '{}_{}_{}_{}_{}_{}_epoch_{}_dev_{}_test_{}.pth.tar'.format(*best_stats)
                # example: bert_dp_TQA_binary_median_20200609_162054_epoch_4_dev_0.6751_test_0.6912.pth.tar

                save_model(model, optimizer, best_dev_epoch, best_dev_epoch_dev_score, 
                           best_dev_epoch_test_score, model_name, args)

                if prev_best_model_name != "":
                    delete_prev_best_model(prev_best_model_name, args.checkpoint_dir)
                prev_best_model_name = model_name  # this model will be deleted in the next time

    # append the best stats
    best_stats.append(best_test_epoch)
    best_stats.append(round(best_test_epoch_dev_score, 4))
    best_stats.append(round(best_test_epoch_test_score, 4))
    best_stats.append(args.seed)
    best_stats_df = pd.DataFrame(columns=best_stats_head_vals)
    best_stats_df.at[0] = best_stats
    best_stats_df.to_csv(best_stats_csv, mode='a', header=False, index=False)


#
#
#
#
#


def eval(dev_iter, model, args):
    n_total_steps = len(dev_iter)
    model.eval()
    dev_loss = 0
    dev_step = 1
    preds = []
    trues = []
    for batch_ids in dev_iter:
        print(">>>>>>>>> Dev step {} / {}".format(dev_step, len(dev_iter)), end="\r")
        input_ids = batch_ids[0].to(torch.device(args.device))
        att_masks = batch_ids[1].to(torch.device(args.device))
        labels = batch_ids[2].to(torch.device(args.device))

        # forward pass
        with torch.no_grad():
            loss, logits = model(input_ids, attention_mask=att_masks, labels=labels)
        dev_loss += loss.item()

        # record preds, trues
        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)
        dev_step += 1

    print("\n")
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
    test_step = 1
    preds = []
    trues = []
    print()
    for batch_ids in test_iter:
        print(">>>>>>>>> Test step {} / {}".format(test_step, len(test_iter)), end="\r")
        input_ids = batch_ids[0].to(torch.device(args.device))
        att_masks = batch_ids[1].to(torch.device(args.device))
        labels = batch_ids[2].to(torch.device(args.device))

        # forward pass
        with torch.no_grad():
            loss, logits = model(input_ids, attention_mask=att_masks, labels=labels)
        test_loss += loss.item()

        # record preds, trues
        _pred = logits.cpu().data.numpy()
        preds.append(_pred)
        _label = labels.cpu().data.numpy()
        trues.append(_label)
        test_step += 1

    print("\n")
    test_loss = test_loss / n_total_steps
    return trues, preds, test_loss


#
#
#
#
#


def calculate_metrics(label, pred, args):
    pred_class = np.concatenate([np.argmax(numarray, axis=1) for numarray in pred]).ravel()
    label_class = np.concatenate([numarray for numarray in label]).ravel()

    logging.info('Expected:  {}'.format(label_class[:20]))
    logging.info('Predicted: {}\n'.format(pred_class[:20]))
    print(classification_report(label_class, pred_class))
    print2logfile('Expected:  {}'.format(label_class[:20]), args)
    print2logfile('Predicted: {}'.format(pred_class[:20]), args)
    print2logfile(classification_report(label_class, pred_class), args)
    acc = accuracy_score(label_class, pred_class)
    f1 = f1_score(label_class, pred_class, average='weighted')
    recall = recall_score(label_class, pred_class, average='weighted')
    prec = precision_score(label_class, pred_class, average='weighted')

    return acc, f1, recall, prec

#
#
#
#
#


def save_model(model, optimizer, epoch, dev_score, test_score, model_name, args):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)

    save_path = os.path.join(args.checkpoint_dir, model_name)

    torch.save({
        'epoch': epoch,
        'dev_score': dev_score,
        'test_score': test_score,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'seed': args.seed,
        'lr': args.lr,
        'device': args.device,
        'model': args.model,
        'labels': args.labels,
        'num_labels': args.num_labels,
        'sequence': args.sequence,
        'crop': args.crop,
        'max_len': args.MAX_LEN
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

#
#
#
#
#

def delete_prev_best_model(prev_best_model_name, checkpoint_dir):
    prev_best_model_path = os.path.join(checkpoint_dir, prev_best_model_name)
    if os.path.exists(prev_best_model_path):
        os.remove(prev_best_model_path)
        logging.info('Previous best model {} is deleted.'.format(prev_best_model_name))