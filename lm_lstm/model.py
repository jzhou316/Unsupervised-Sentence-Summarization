# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 2018

@author: zjw
"""
import torch
import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, padid=1, max_norm=None, tieweights=False):
        super(RNNModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.padid = padid
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=self.padid, max_norm=None)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout)
        self.drop = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=True)

        self.init_weight(0.5)
        
        # tie weights
        if tieweights:
            self.proj.weight = self.embedding.weight
    
    def init_weight(self, initrange=0.1):
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
#         nn.init.uniform_(self.proj.weight, -initrange, initrange)
        nn.init.orthogonal_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0)
        
    def forward(self, batch_text, hn, subvocab=None, return_prob=False):
        embed = self.embedding(batch_text)    # size: (seq_len, batch_size, embed_size)
        output, hn = self.lstm(embed, hn)     # output size: (seq_len, batch_size, hidden_size)
        output = self.drop(output)            # hn = (hn, cn), each with size: (num_layers, batch, hidden_size)
        if isinstance(subvocab, list):
            subvocab = torch.LongTensor(subvocab, device=output.device)
        output = self.proj(output) if subvocab is None else nn.functional.linear(output, self.proj.weight[subvocab, :], self.proj.bias[subvocab])
        if return_prob:
            output = nn.functional.softmax(output, dim=-1)
        # detach last hidden and cell states to truncate the computational graph for BPTT.
        hn = tuple(map(lambda x: x.detach(), hn))
        return output, hn
 
    def score_textseq(self, text, vocab, hn=None, size_average=True):
        """
        Output the log-likelihood of a text sequence.
        """
        if isinstance(text, str):
            text = text.split()
        textid = next(self.parameters()).new_tensor([vocab.stoi[w] for w in text], dtype=torch.long)
        with torch.no_grad():
            self.eval()
            model_output, hn = self(textid.unsqueeze(1), hn)
            self.train() 
        ll = nn.functional.cross_entropy(model_output[:-1, 0, :], textid[1:], ignore_index=self.padid,
                                  reduction='elementwise_mean' if size_average else 'sum')
        ll = -ll.item()
        return ll
    
    def score_nexttoken(self, text, vocab, hn=None):
        """
        Output the predictive probabilities of the next token given a text sequence.
        """
        if isinstance(text, str):
            text = text.split()
        textid = next(self.parameters()).new_tensor([vocab.stoi[w] for w in text], dtype=torch.long)
        with torch.no_grad():
            self.eval()
            model_output, hn = self(textid.unsqueeze(1), hn, return_prob=True)
            self.train()
        
        return model_output[-1, 0, :]
        
