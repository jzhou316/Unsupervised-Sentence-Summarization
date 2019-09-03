# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 23:25:26 2018

@author: zjw
"""

import torchtext


def loadPTB(root='E:/NLP/LM/data', batch_size=64, bptt_len=32, device=None, **kwargs):
    """
    Load the Penn Treebank dataset. Download if not existing.
    """
    TEXT = torchtext.data.Field(lower=True)
    train, val, test = torchtext.datasets.PennTreebank.splits(root=root, text_field=TEXT)
    TEXT.build_vocab(train, **kwargs)    # could include: max_size, min_freq, vectors
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test),
                                                  batch_size=batch_size,
                                                  bptt_len=bptt_len,
                                                  device=device,
                                                  repeat=False)
    
    return TEXT, train_iter, val_iter, test_iter


def loadWiki2(root='E:/NLP/LM/data', batch_size=64, bptt_len=32, device=None, **kwargs):
    """
    Load the WikiText2 dataset. Download if not existing.
    """
    TEXT = torchtext.data.Field(lower=True)
    train, val, test = torchtext.datasets.WikiText2.splits(root=root, text_field=TEXT)
    TEXT.build_vocab(train, **kwargs)
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test),
                                                  batch_size=batch_size,
                                                  bptt_len=bptt_len,
                                                  device=device,
                                                  repeat=False)
    
    return TEXT, train_iter, val_iter, test_iter


def loadLMdata(path='E:/NLP/LM/data/penn-tree-bank-small',
               train='ptb.train.5k.txt',
               val='ptb.valid.txt',
               test='ptb.test.txt',
               batch_size=64,
               bptt_len=32,
               device=None, **kwargs):
    """
    Load a dataset for LM training. The dataset should exist already.
    """
    TEXT = torchtext.data.Field(lower=True)
    train, val, test = torchtext.datasets.LanguageModelingDataset.splits(path=path,
                                                  train=train,
                                                  validation=val,
                                                  test=test,
                                                  text_field=TEXT)
    TEXT.build_vocab(train, val, test, **kwargs)
    train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits((train, val, test),
                                                  batch_size=batch_size,
                                                  bptt_len=bptt_len,
                                                  device=device,
                                                  repeat=False)
    
    return TEXT, train_iter, val_iter, test_iter