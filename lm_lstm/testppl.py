# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 2019

@author: zjw
"""
import torch
from dataload import loadPTB, loadWiki2, loadLMdata
from model import RNNModel
from train import validating
# from train_sharding import training, validating

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

vocabsavepath = './models/vocabTle.pkl'

def parse_args():
    parser = argparse.ArgumentParser(description='Training an LSTM language model.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--devid', type=int, default=-1, help='single device id; -1 for CPU')
    group.add_argument('--devids', type=str, default='off', help='multiple device ids for data parallel; use comma to separate, e.g. 0, 1, 2')
#     parser.add_argument('--devid', type=int, default=-1, help='device id; -1 for CPU')
##    parser.add_argument('--modelfile', type=str, default='model', help='file name of the model, without .py')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
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
    parser.add_argument('--model', type=str, default='off', help='a trained model to start with')
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


# random.seed(args.seed)        # this has no impact on the current model training
torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False

# print('-' * 30)
# print(time.ctime())

########## load the dataset
print('-' * 30)
print('Loading data ...')

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
print(f'Vocabulary size: {len(TEXT.vocab)}')
print('Complete!')
print('-' * 30)

########## define the model
LMModel = torch.load(args.model).cpu()
'''
LMModel_start = torch.load(args.start_model).cpu()
# Note: watch out if the model class has different methods from the loaded one to start with !!!
LMModel = RNNModel(vocab_size=vocab_size, 
                   embed_size=LMModel_start.embedsz,
                   hidden_size=LMModel_start.hiddensz, 
                   num_layers=LMModel_start.numlayers,
                   dropout=LMModel_start.dropout, 
                   padid=LMModel_start.padid, 
                   tieweights=LMModel_start.tieweights)
LMModel.load_state_dict(LMModel_start.state_dict())
'''    

# LMModel = torch.load(args.save).cpu()

model_size = sum(p.nelement() for p in LMModel.parameters())

print('-' * 30)
print(f'Model tatal parameters: {model_size}')
print('-' * 30)

if torch.cuda.is_available() and cuda_device is not 'cpu':
    LMModel = LMModel.cuda(cuda_device)

LMModel_parallel = None
if torch.cuda.is_available() and args.devids is not 'off':
    LMModel_parallel = torch.nn.DataParallel(LMModel, device_ids=device_ids, output_device=output_device, dim=1)    
                                                             # .cuda() is necessary if LMModel was not on any GPU device
#     LMModel_parallel._modules['module'].lstm.flatten_parameters()
'''
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
'''
#         print('-' * 30)
#         print('Loading saved optimizer states.')
#         print('-' * 30)

######### test the trained model
test_ppl = validating(test_iter, LMModel, LMModel_parallel=LMModel_parallel)

print('-' * 30)
print('Test ppl: %f' % test_ppl)
print('-' * 30)

