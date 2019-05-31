'''
Sequentially embed word tokens into GPT-2 (last-layer) hidden state vectors, with model internal states from the past saved.
Note that GPT-2 uses BPE encodings for its vocabulary, so each word type will have multiple BPE units of variable length.

Based on the library pytorch_pretrained_bert.
'''

import torch
import torch.nn as nn
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model

import logging
logging.basicConfig(level=logging.INFO)


class GPT2Embedder(nn.Module):
    def __init__(self, cuda_device=-1):
        super(GPT2Embedder, self).__init__()
        
        self.cuda_device = 'cpu' if cuda_device == -1 else f'cuda:{cuda_device}'
        
        # Load pre-trained model tokenizer (vocabulary)
        self.enc = GPT2Tokenizer.from_pretrained('gpt2')
        # Load pre-trained model (weights)
        self.model = GPT2Model.from_pretrained('gpt2')
        
        self.model.to(self.cuda_device)
        self.model.eval()        # we only use the evaluation mode of the pretrained model
    
        self._bos_id = self.enc.encoder['<|endoftext|>']
        self._bos_past = None
        
    @property
    def bos_past(self):
        if self._bos_past is not None:
            return self._bos_past
        else:
            with torch.no_grad():
                _, self._bos_past = self.model(torch.tensor([[self._bos_id]], device=self.cuda_device), past=None)
            return self._bos_past
        
    def embed_sentence(self, sentence, add_bos=False, add_eos=False, bpe2word='last', initial_state=None):
        '''
        Compute the GPT-2 embeddings for a single tokenized sentence.
        
        Input:
            sentence (List[str]): tokenized sentence
            add_bos (bool): whether to add begin of sentence token '<|endoftext|>'
            add_eos (bool): whetehr to add end of sentenc token '<|endoftext|>' (currently not used)
            bpe2word (str): how to turn the BPE vectors into word vectors;
                 'last': last hidden state; 'avg': average hidden state.
            initial_state (List[torch.Tensor]): GPT-2 internal states for the past
            
        Output:
            embeddings (torch.Tensor): GPT-2 vectors for the sentence, size (len(sentence), 768)
            states (List[torch.Tensor]): GPT-2 internal states for the past, a list of length 12 (for 12 layers)
        '''
        assert isinstance(sentence, list), 'input "sentence" should be a list of word types.'
        assert bpe2word in ['last', 'avg']
        
        if add_bos:
            # initial_state is not used when 'add_bos' is True
            past = self.bos_past
        else:
            past = initial_state
            
        if past is None:
            bos_sp = ''        # begin of sentence: whether there is a space or not
        else:
            bos_sp = ' '
        
        for i, w in enumerate(sentence):
            if i == 0:
                bpe_units = torch.tensor([self.enc.encode(bos_sp + w)], device=self.cuda_device)
                with torch.no_grad():
                    vec, past = self.model(bpe_units, past=past)
            else:
                bpe_units = torch.tensor([self.enc.encode(' ' + w)], device=self.cuda_device)
                with torch.no_grad():
                    vec, past = self.model(bpe_units, past=past)
        
            if bpe2word == 'last':
                vec = vec[:, -1, :]
            elif bpe2word == 'avg':
                vec = vec.mean(dim=1)
            else:
                raise ValueError
    
            embeddings = vec if i == 0 else torch.cat([embeddings, vec], dim=0)
            
        return embeddings, past
        
    def embed_words(self, words, add_bos=False, add_eos=False, bpe2word='last', initial_state=None):
        '''
        Compute the GPT-2 embeddings for a list of words.
        The challenge is that these words might have BPE encodings of different lengths, so we need to pad for a batch and then
        correctly index out the embeddings and internal states at right positions.
        
        Input:
            words (List[str]): a list of words
            add_bos (bool): whether to add begin of sentence token '<|endoftext|>'
            add_eos (bool): whetehr to add end of sentenc token '<|endoftext|>' (currently not used)
            bpe2word (str): how to turn the BPE vectors into word vectors;
                 'last': last hidden state; 'avg': average hidden state.
            initial_state (List[torch.Tensor]): GPT-2 internal states for the past
            
        Output:
            embeddings (torch.Tensor): GPT-2 vectors for the words, size (len(words), 768)
            states (List[List[torch.Tensor]]): GPT-2 internal states for the past, a list of length len(words)
        '''
        assert isinstance(words, list), 'input "words" should be a list of candidate word types for the next step.'
        assert bpe2word in ['last', 'avg']
        
        if add_bos:
            # initial_state is not used when 'add_bos' is True
            past = self.bos_past
        else:
            past = initial_state
            
        if past is None:
            bos_sp = ''        # begin of sentence: whether there is a space or not
        else:
            bos_sp = ' '
        
        n = len(words)
        bpe_list = [self.enc.encode(bos_sp + w) for w in words]
        bpe_lens = [len(b) for b in bpe_list]
        
        ## padding to for a batch
        padding = 0
        max_seqlen = max(bpe_lens)
        bpe_list = [b + [padding] * (max_seqlen - l) for b, l in zip(bpe_list, bpe_lens)]
        bpe_padded = torch.tensor(bpe_list, device=self.cuda_device)        # size (n, max_seqlen)
        if past is not None:
            past_seqlen = past[0].size(3)
            past = [p.expand(-1, n, -1, -1, -1) for p in past]        # same past internal states for every word in the batch
        else:
            past_seqlen = 0
        
        ## run GPT-2 model
        with torch.no_grad():
            hid, mid = self.model(bpe_padded, past=past)
        
        ## extract the hidden states of words through indexing
        if bpe2word == 'last':
            # method 1: torch.gather
            index = torch.tensor(bpe_lens, device=self.cuda_device).reshape(n, 1, 1).expand(-1, -1, hid.size(2)) - 1
            embeddings = torch.gather(hid, 1, index).squeeze(1)
            # method 2: for loop
#             embeddings = hid.new_zeros(n, hid.size(2))
#             for i in range(n):
#                 embeddings[i] = hid[i, bpe_lens[i] - 1]
        elif bpe2word == 'avg':
            a = torch.arange(max_seqlen, device=self.cuda_device).view(1, -1).expand(n, -1)
            b = torch.tensor(bpe_lens, device=self.cuda_device).view(-1, 1)
            mask = a >= b        # size (n, max_seqlen)
            hid[mask] = 0        # mask out the padded position embeddings
            embeddings = hid.sum(dim=1) / b.float()
        else:
            raise ValueError
        
        ## index out the internal states
        states = torch.cat(mid, dim=0)        # size (2 * 12, n, 12, past_seqlen + max_seqlen, 64)
        states = torch.split(states, 1, dim=1)        # list of length n
        states = [torch.chunk(s.index_select(3, torch.arange(past_seqlen + l, device=self.cuda_device)), 12, dim=0)
                  for s, l in zip(states, bpe_lens)]
        
        return embeddings, states

