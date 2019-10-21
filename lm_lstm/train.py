# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 2018

@author: zjw
"""
import torch
import torch.nn as nn
import math
import time
import os
from utils import logging, timeSince, rand_subvocab


def training(train_iter, val_iter, num_epoch, LMModel, optimizer, scheduler, grad_max_norm=None, savepath='./LMModel.pth', LMModel_parallel=None, f_log=None, subvocab_size=None):
    criterion = nn.CrossEntropyLoss(ignore_index=LMModel.padid, reduction='sum')
    best_val_ppl = None
    last_epoch = scheduler.last_epoch
    LMModel.train()
    start = time.time()
    for epoch in range(last_epoch + 1, last_epoch + 1 + num_epoch):
        train_iter.init_epoch()
        loss_total = 0
        num_token_passed = 0
        hn = None
        for batch in train_iter:
            # calculate sub-vocabulary
##            subvocab = rand_subvocab(batch, LMModel.vocab_size, subvocab_size)
            subvocab = None
            # update parameters
            optimizer.zero_grad()
            if LMModel_parallel is None:
                output, hn = LMModel(batch.text, hn if hn is not None else None, subvocab=subvocab)
            else:
                output, hn = LMModel_parallel(batch.text, hn if hn is not None else None, subvocab=subvocab.numpy().tolist() if subvocab is not None else None)
            if subvocab is not None:
                target_subids = batch.target.new_tensor([(subvocab == x).nonzero().item() for x in batch.target.view(-1).cpu()], dtype=torch.long)
            loss = criterion(output.view(-1, output.size(2)), batch.target.view(-1)) if subvocab is None else \
                   criterion(output.view(-1, output.size(2)), target_subids)
            loss.backward()

            if grad_max_norm:
                nn.utils.clip_grad_norm_(LMModel.parameters(), grad_max_norm)
            
            # calculate perplexity
            loss_total += float(loss)        # do not accumulate history accross training loop
            num_token_passed += (torch.numel(batch.target) -
                                 torch.sum(batch.target == LMModel.padid)).item()
                                             # do not count the '<pad>', which could only exist
                                             # at the end of the last batch
            loss_avg = loss_total / num_token_passed
            ppl = math.exp(loss_avg)
 
            optimizer.step()

            # print information
            if train_iter.iterations % 50 == 0 or train_iter.iterations == len(train_iter):
                logging('Epoch %d / %d, iteration %d / %d, ppl: %f (time elasped %s)'
                      %(epoch + 1, last_epoch + 1 + num_epoch, train_iter.iterations, len(train_iter), ppl, timeSince(start)), f_log=f_log)
            
        # calculation ppl on validation set
        val_ppl = validating(val_iter, LMModel, LMModel_parallel=LMModel_parallel, f_log=f_log)
        LMModel.train()
        logging('-' * 30, f_log=f_log)
        logging('Validating ppl: %f' % val_ppl, f_log=f_log)
        logging('-' * 30, f_log=f_log)
        
        scheduler.step(val_ppl)
        
        # save the model if the validation ppl is the best so far
        if not best_val_ppl or val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(LMModel, savepath)
            torch.save(optimizer.state_dict(), os.path.splitext(savepath)[0] + '_optstate.pth')
            torch.save(scheduler.state_dict(), os.path.splitext(savepath)[0] + '_schstate.pth')
            torch.save({'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state_all()}, os.path.splitext(savepath)[0] + '_rngstate.pth')
            logging(f'Current model (after epoch {epoch+1}) saved to {savepath} (along with optimizer state dictionary & scheduler state dictionary & rng states)', f_log=f_log)
            logging('-' * 30, f_log=f_log)
        
    return ppl, val_ppl


def validating(val_iter, LMModel, LMModel_parallel=None, f_log=None):
    criterion = nn.CrossEntropyLoss(ignore_index=LMModel.padid, reduction='sum')
    LMModel.eval()
    with torch.no_grad():
        val_iter.init_epoch()
        loss_total = 0
        num_token_passed = 0
        hn = None
        for batch in val_iter:
            if LMModel_parallel is None:
                output, hn = LMModel(batch.text, hn if hn is not None else None)
            else:
                output, hn = LMModel_parallel(batch.text, hn if hn is not None else None)
            loss = criterion(output.view(-1, output.size(2)), batch.target.view(-1))
            loss_total += float(loss)
            num_token_passed += torch.sum(batch.target.ne(LMModel.padid)).item()
            
        ppl = math.exp(loss_total / num_token_passed)
    return ppl
