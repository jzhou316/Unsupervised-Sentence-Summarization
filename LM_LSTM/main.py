# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 2018

@author: zjw
"""
import torch
import torch.optim as optim
from dataload import loadPTB, loadWiki2, loadLMdata
from model import RNNModel
from train import training, validating
# from train_sharding import training, validating
from utils import logging

import os
import sys
# import random
import argparse
import time
import importlib
import pickle


########## set up parameters
# data
data_src = 'ptb'
# on MicroSoft Azure
# data_root = '/media/work/LM/data'
# userdata_path = '/media/work/LM/data/Giga-sum'        # .../penn-treebank-small
# on Harvard Odyssey Cluster
data_root = '/n/rush_lab/users/jzhou/LM/data'
userdata_path = '/n/rush_lab/users/jzhou/LM/data/Giga-sum'        # .../penn-treebank-small
userdata_train = 'train.title.txt'
userdata_val = 'valid.title.filter.txt'
userdata_test = 'task1_ref0_unk.txt'
batch_size = 128
bptt_len = 32
# model
embed_size = 512    # 1024
hidden_size = 512    # 1024
num_layers = 2
dropout = 0.5
tieweights = 0     # 0 for False, 1 for True
# optimization
learning_rate = 0.01
momentum = 0.9
weight_decay = 1e-4
grad_max_norm = 120    # 1024, 0.01 ---> 120
shard_size = 64
##subvocab_size = 0
#learning_rate = 0.001
#grad_max_norm = None
num_epochs = 50

vocabsavepath = './models/vocabTle.pkl'
savepath = './models/Tle_LSTM.pth'
#savepath = '/media/work/LM/LMModel.pth'
#savepath = '/n/rush_lab/users/jzhou/LM/LMModel.pth'


def parse_args():
    parser = argparse.ArgumentParser(description='Training an LSTM language model.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--devid', type=int, default=-1, help='single device id; -1 for CPU')
    group.add_argument('--devids', type=str, default='off', help='multiple device ids for data parallel; use comma to separate, e.g. 0, 1, 2')
#     parser.add_argument('--devid', type=int, default=-1, help='device id; -1 for CPU')
##    parser.add_argument('--modelfile', type=str, default='model', help='file name of the model, without .py')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--logmode', type=str, default='w', help='logging file mode')
    # data loading
    parser.add_argument('--data_src', type=str, default=data_src, choices=['ptb', 'wiki2', 'user'], help='data source')
    parser.add_argument('--data_root', type=str, default=data_root, help='root path for PTB/Wiki2 dataset path')
    parser.add_argument('--userdata_path', type=str, default=userdata_path, help='user data path')
    parser.add_argument('--userdata_train', type=str, default=userdata_train, help='user data training set file name')
    parser.add_argument('--userdata_val', type=str, default=userdata_val, help='user data validating set file name')
    parser.add_argument('--userdata_test', type=str, default=userdata_test, help='user data testing set file name')
    parser.add_argument('--bptt', type=int, default=bptt_len, help='bptt length')
    parser.add_argument('--bsz', type=int, default=batch_size, help='batch size')
    parser.add_argument('--vocabsave', type=str, default=vocabsavepath, help='file path to save the vocabulary object')
    # model
    parser.add_argument('--embedsz', type=int, default=embed_size, help='word embedding size')
    parser.add_argument('--hiddensz', type=int, default=hidden_size, help='hidden state size')
    parser.add_argument('--numlayers', type=int, default=num_layers, help='number of layers')
    parser.add_argument('--dropout', type=float, default=dropout, help='dropout probability')
#     parser.add_argument('--tieweights', help='whether to tie input and output embedding weights', action='store_true')
    parser.add_argument('--tieweights', type=int, default=tieweights, help='whether to tie input and output embedding weights')
    parser.add_argument('--start_model', type=str, default='off', help='a trained model to start with')
    # optimization
    parser.add_argument('--optim', type=str, default='SGD', choices=['SGD', 'Adam'], help='optimization algorithm')
    parser.add_argument('--lr', type=float, default=learning_rate, help='learning rate')
    parser.add_argument('--momentum', type=float, default=momentum, help='momentum for SGD')
    parser.add_argument('--wd', type=float, default=weight_decay, help='weight decay (L2 penalty)')
    parser.add_argument('--gradclip', type=float, default=grad_max_norm, help='gradient norm clip')
##    parser.add_argument('--shardsz', type=int, default=shard_size, help='shard size for mixture of softmax output layer')
##    parser.add_argument('--subvocabsz', type=int, default=subvocab_size, help='sub-vocabulary size for training on large corpus')
    parser.add_argument('--epochs', type=int, default=num_epochs, help='number of training epochs')
    parser.add_argument('--save', type=str, default=savepath, help='file path to save the best model')
    args = parser.parse_args()
    return args

args = parse_args()


## RNNModel = importlib.import_module(args.modelfile).RNNModel

cuda_device = 'cpu' if args.devid == -1 else f'cuda:{args.devid}'
if args.devids is not 'off':
    device_ids = list(map(int, args.devids.split(',')))
    output_device = device_ids[0]
    cuda_device = f'cuda:{output_device}'

# if os.name == 'nt':
#     # run on my personal windows computer with cpu
#     cuda_device = None
#     root = 'E:/NLP/LM/data'
#     path = 'E:/NLP/LM/data/penn-treebank-small'
# elif os.name == 'posix':
#     # run on Harvard Odyssey cluster
#     cuda_device = 'cuda:0'
# #     root = '/n/rush_lab/users/jzhou/LM/data'
# #     path = '/n/rush_lab/users/jzhou/LM/data/penn-treebank-small'
#     root = '/media/work/LM/data'
#     path = '/media/work/LM/data/penn-treebank'
# #     path = '/media/work/LM/data/Giga-sum'
# #     train = 'train.title.txt'
# #     val = 'valid.title.filter.txt'
# #     test = 'task1_ref0.txt'

log_file = os.path.splitext(args.save)[0] + '.log'
f_log = open(log_file, args.logmode)

logging('python ' + ' '.join(sys.argv), f_log=f_log)

logging('-' * 30, f_log=f_log)
logging(time.ctime(), f_log=f_log)


# random.seed(args.seed)        # this has no impact on the current model training
torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

# print('-' * 30)
# print(time.ctime())

########## load the dataset
logging('-' * 30, f_log=f_log)
logging('Loading data ...', f_log=f_log)

# print('-' * 30)
# print('Loading data ...')

if args.data_src == 'ptb':
    TEXT, train_iter, val_iter, test_iter = loadPTB(root=args.data_root,
                                                    batch_size=args.bsz,
                                                    bptt_len=args.bptt,
                                                    device=cuda_device)
elif args.data_src == 'wiki2':
    TEXT, train_iter, val_iter, test_iter = loadWiki2(root=args.data_root,
                                                      batch_size=args.bsz,
                                                      bptt_len=args.bptt,
                                                      device=cuda_device)
elif args.data_src == 'user':
    TEXT, train_iter, val_iter, test_iter = loadLMdata(path=args.userdata_path,
                                                       train=args.userdata_train,
                                                       val=args.userdata_val,
                                                       test=args.userdata_test,
                                                       batch_size=args.bsz,
                                                       bptt_len=args.bptt,
                                                       device=cuda_device,
                                                       min_freq=5)

padid = TEXT.vocab.stoi['<pad>']
vocab_size = len(TEXT.vocab)

logging(f'Vocab size: {vocab_size}', f_log=f_log)
if not os.path.exists(args.vocabsave):
    pickle.dump(TEXT.vocab, open(args.vocabsave, 'wb'))
    logging(f'Vocabulary object saved to: {args.vocabsave}', f_log=f_log)
else:
    logging(f'Vocabulary object at: {args.vocabsave}', f_log=f_log)
logging('Complete!', f_log=f_log)
logging('-' * 30, f_log=f_log)

##if args.subvocabsz >= vocab_size or args.subvocabsz == 0:
##    args.subvocabsz = None

# print('Complete!')
# print('-' * 30)

########## define the model and optimizer

if args.start_model is 'off':
    LMModel = RNNModel(vocab_size=vocab_size, 
                       embed_size=args.embedsz,
                       hidden_size=args.hiddensz, 
                       num_layers=args.numlayers,
                       dropout=args.dropout, 
                       padid=padid, 
                       tieweights=args.tieweights)
else:
    LMModel_start = torch.load(args.start_model).cpu()
    # Note: watch out if the model class has different methods from the loaded one to start with !!!
    LMModel = RNNModel(vocab_size=vocab_size, 
                       embed_size=args.embedsz,
                       hidden_size=args.hiddensz, 
                       num_layers=args.numlayers,
                       dropout=args.dropout, 
                       padid=padid, 
                       tieweights=args.tieweights)
    LMModel.load_state_dict(LMModel_start.state_dict())
    

# LMModel = torch.load(args.save).cpu()

model_size = sum(p.nelement() for p in LMModel.parameters())
logging('-' * 30, f_log=f_log)
logging(f'Model tatal parameters: {model_size}', f_log=f_log)
logging('-' * 30, f_log=f_log)

# print('-' * 30)
# print(f'Model tatal parameters: {model_size}')
# print('-' * 30)

if torch.cuda.is_available() and cuda_device is not 'cpu':
    LMModel = LMModel.cuda(cuda_device)

LMModel_parallel = None
if torch.cuda.is_available() and args.devids is not 'off':
    LMModel_parallel = torch.nn.DataParallel(LMModel, device_ids=device_ids, output_device=output_device, dim=1)    
                                                             # .cuda() is necessary if LMModel was not on any GPU device
#     LMModel_parallel._modules['module'].lstm.flatten_parameters()

if args.optim == 'SGD':
    optimizer = optim.SGD(LMModel.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
elif args.optim == 'Adam':
    optimizer = optim.Adam(LMModel.parameters(), lr=args.lr, weight_decay=args.wd)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1)

if args.start_model is not 'off':
    start_model_optstate_path = os.path.splitext(args.start_model)[0] + '_optstate.pth'
    start_model_schstate_path = os.path.splitext(args.start_model)[0] + '_schstate.pth'
    if os.path.exists(start_model_optstate_path):
        optimizer.load_state_dict(torch.load(start_model_optstate_path))
        logging('-' * 30, f_log=f_log)
        logging('Loading saved optimizer states.', f_log=f_log)
        logging('-' * 30, f_log=f_log)

    if os.path.exists(start_model_schstate_path):
        scheduler.load_state_dict(torch.load(start_model_schstate_path))
        logging('-' * 30, f_log=f_log)
        logging('Loading saved scheduler states.', f_log=f_log)
        logging('-' * 30, f_log=f_log)

#         print('-' * 30)
#         print('Loading saved optimizer states.')
#         print('-' * 30)

########## traing the model
if args.start_model is not 'off':
    start_model_rngstate_path = os.path.splitext(args.start_model)[0] + '_rngstate.pth'
    if os.path.exists(start_model_rngstate_path):
        torch.set_rng_state(torch.load(start_model_rngstate_path)['torch_rng_state'])
        torch.cuda.set_rng_state_all(torch.load(start_model_rngstate_path)['cuda_rng_state'])
        logging('-' * 30, f_log=f_log)
        logging('Loading saved rng states.', f_log=f_log)
        logging('-' * 30, f_log=f_log)

train_ppl, val_ppl = training(train_iter, val_iter, args.epochs,
                              LMModel,
                              optimizer,
                              scheduler,
                              args.gradclip,
                              args.save,
##                              shard_size=args.shardsz,
                              LMModel_parallel=LMModel_parallel,
                              f_log=f_log)
##                              subvocab_size=args.subvocabsz)

######### test the trained model
##test_ppl = validating(test_iter, LMModel, shard_size=args.shardsz, LMModel_parallel=LMModel_parallel, f_log=f_log)
test_ppl = validating(test_iter, LMModel, LMModel_parallel=LMModel_parallel, f_log=f_log)
logging('-' * 30, f_log=f_log)
logging('Test ppl: %f' % test_ppl, f_log=f_log)
logging('-' * 30, f_log=f_log)

f_log.close()

# print('-' * 30)
# print('Test ppl: %f' % test_ppl)
# print('-' * 30)
