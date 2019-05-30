import numpy as np
import torch
import time
from utils import timeSince
# from tqdm import tqdm

from sim_token_match import OneTokenMatch


def pickElmoForwardLayer(embedding, elmo_layer='avg'):
    """
    Given a forward only ELMo embedding vector of size (3, #words, 512), pick up the layer
    """
    assert elmo_layer in ['top', 'mid', 'bot', 'avg', 'cat']

    if elmo_layer == 'top':
        embedding = embedding[2]
    elif elmo_layer == 'mid':
        embedding = embedding[1]
    elif elmo_layer == 'bot':
        embedding = embedding[0]
    elif elmo_layer == 'avg':
        if isinstance(embedding, np.ndarray):
            embedding = np.average(embedding, axis=0)
        elif isinstance(embedding, torch.Tensor):
            embedding = torch.mean(embedding, dim=0)
    elif elmo_layer == 'cat':
        if isinstance(embedding, np.ndarray):
            embedding = np.reshape(embedding.transpose(1, 0, 2), (-1, embedding.shape[0] * embedding.shape[2]))   # concat 3 layers, bottom first
        elif isinstance(embedding, torch.Tensor):
            embedding = embedding.transpose(0, 1).reshape(-1, embedding.size(0) * embedding.size(2))

    return embedding


def simScoreNext(template_vec, word_list, ee, batch_size=1024, prevs_state=None,
                 prevs_align=None, normalized=True, elmo_layer='avg'):
    '''
    Score the next tokens based on sentence level similarity, with previous alignment fixed.
    
    Input:
        template_vec: template sentence ELMo vectors.
        word_list: a list of next candidate words.
        ee: a ``ElmoEmbedderForward`` class.
        batch_size: for ee to use.
        prevs_state: previous hidden states.
        prevs_align: aligning location for the last word in the sequence.
                      If provided, monotonicity is required.
        normalized: whether to use normalized dot product (cosine similarity) for token similarity calculation.
        elmo_layer: ELMo layer to use.
    Output:
        scores: unsorted one-token similarity scores, torch.Tensor.
        indices: matched indices in template_vec for each token, torch.LongTensor.
        states: corresponding ELMo forward lstm hidden states, List.
    '''
    sentences = [[w] for w in word_list]
    src_vec = pickElmoForwardLayer(template_vec, elmo_layer)
    if prevs_state is None:
        assert prevs_align is None, 'Nothing should be passed in when no history.'
        # beginning of sentence, the first token
        embeddings_and_states = ee.embed_sentences(sentences, add_bos=True, batch_size=batch_size)                
    else:
        # in the middle of sentence, sequential update
#         start = time.time()
        embeddings_and_states = ee.embed_sentences(sentences, initial_state=prevs_state, batch_size=batch_size)
#         print('ELMo embedding: ' + timeSince(start))
    
    embeddings, states = zip(*embeddings_and_states)           # this returns two tuples
        
    scores = []
    indices = []
    print('Calculating similarities ---')
#     start = time.time()
    embeddings = [pickElmoForwardLayer(vec, elmo_layer) for vec in embeddings]
    scores, indices = OneTokenMatch(src_vec, embeddings, normalized=normalized, starting_loc=prevs_align)
#     print('Similarities: ' + timeSince(start))
        
    return scores, indices, list(states)


def simScoreNext_GPT2(template_vec,
                      word_list,
                      tokenizer,
                      model,
                      bpe2word='last',
                      prevs_state=None,
                      prevs_align=None,
                      normalized=True):
    '''
    Score the next tokens based on sentence level similarity, with previous alignment fixed.
    In particular, this function uses GPT-2 to embed the sentences/candidate words:
    - Calculate the embeddings for each candidate word using pretrained GPT-2 model, given the previous hidden states
    - Calculate best alignment positions and similarity scores for each word
    
    Note:
        - GPT-2 uses BPE tokenizer, so each word may be splitted into several different units
        
    Input:
        template_vec (torch.Tensor): template sentence GPT-2 embedding vectors
        word_list (list): a list of next candidate words
        tokenizer (pytorch_pretrained_bert.tokenization_gpt2.GPT2Tokenizer): GPT-2 tokenizer
        model (pytorch_pretrained_bert.modeling_gpt2.GPT2Model): GPT-2 Model
        bpe2word (str): how to turn the BPE vectors into word vectors.
            'last': last hidden state; 'avg': average hidden state.
        prevs_state (list[torch.Tensor]): previous hidden states for the GPT-2 model
        prevs_align (int): aligning location for the last word in the sequence.
            If provided, monotonicity is required.
        normalized (bool): whether to use normalized dot product (cosine similarity) for token similarity calculation
    
    Output:
        scores (torch.Tensor): unsorted one-token similarity scores
        indices (torch.LongTensor): matched indices in template_vec for each token
        states (list): corresponding GPT-2 hidden states
    '''
    assert bpe2word in ['last', 'avg']
    
    device = next(model.parameters()).device
    model.eval()
    
    ## bpe encoding of the candidate words
    if prevs_state is None:
        # beginning of sentence, the first token
        assert prevs_align is None, 'Nothing should be passed in when no history.'
        bpe_list = [tokenizer.encode(w) for w in word_list]
    else:
        # in the middle of a sentence, sequential update
        prevs_state = [p.expand(-1, len(word_list), -1, -1, -1) for p in prevs_state]    # batch
        bpe_list = [tokenizer.encode(' ' + w) for w in word_list]        

    bpe_lens = [len(b) for b in bpe_list]

    ## padding to form a batch
    padding = 0
    size = max(bpe_lens)
    bpe_list = [b + [padding] * (size - l) for b, l in zip(bpe_list, bpe_lens)]
    bpe_padded = torch.tensor(bpe_list, device=device)        # size (len(word_list), max(bpe_lens))
    
    ## run GPT-2 model
    start = time.time()
    with torch.no_grad():
        hid, mid = model(bpe_padded, past=prevs_state)        # hid size: (len(word_list), max(bpe_lens), 768)
    print('run GPT-2: ' + timeSince(start))
    
    ## extract the hidden states of words through indexing
    start = time.time()
    if bpe2word == 'last':
#         index = torch.tensor(bpe_lens, device=device).reshape(hid.size(0), 1, 1).expand(-1, -1, hid.size(2)) - 1
#         print(f'calculate index: {time.time()-start}')
#         embeddings = torch.gather(hid, 1, index).squeeze(1)
        embeddings = hid.new_zeros(hid.size(0), hid.size(2))
        for i in range(hid.size(0)):
            embeddings[i] = hid[i, bpe_lens[i] - 1]
    elif bpe2word == 'avg':
        a = torch.arange(size, device=device).view(1, -1).expand(hid.size(0), -1)
        b = torch.tensor(bpe_lens, device=device).view(-1, 1)
        mask = a >= b        # size (len(word_list), max(bpe_lens))
        print(f'calculate mask: ' + timeSince(start))
        hid[mask] = 0        # mask out the padded position embeddings
        embeddings = hid.sum(dim=1) / b.float()
    else:
        raise ValueError
    print(f'extract correct embeddings: {time.time()-start}')
    
    ## index out the middle layer states
    start = time.time()
    states = torch.cat(mid, dim=0)        # size (2 * 12, len(word_list), 12, max(bpe_lens), 64)
    states = torch.split(states, 1, dim=1)    # list of size len(word_list)
    states = [torch.chunk(s.index_select(3, torch.arange(l, device=device)), 12, dim=0)
             for s, l in zip(states, bpe_lens)]
    print(f'extract correct states: {time.time()-start}')
    
    scores = []
    indices = []
    print('Calculating similarities ---')
    start = time.time()
    scores, indices = OneTokenMatch(template_vec, embeddings, normalized=normalized, starting_loc=prevs_align)
    print('Similarities: ' + timeSince(start))
        
    return scores, indices, states


"""
def simScoreNext_GPT2(template_vec, 
                      bpe_encoding_grouped,
                      model,
                      bpe2word='last',
                      prevs_state=None, prevs_align=None, normalized=True):
    '''
    Score the next tokens based on sentence level similarity, with previous alignment fixed.
    In particular, this function uses GPT-2 to embed the sentences/candidate words:
    - Calculate the embeddings for each candidate word using pretrained GPT-2 model, given the previous hidden states
    - Calculate best alignment positions and similarity scores for each word
    
    Note:
        - GPT-2 uses BPE tokenizer, so each word may be splitted into several different units
        
    Input:
        template_vec (torch.Tensor): template sentence GPT-2 embedding vectors
        word_list (list): a list of next candidate words
        prevs_state (list[torch.Tensor]): previous hidden states for the GPT-2 model
        tokenizer (pytorch_pretrained_bert.tokenization_gpt2.GPT2Tokenizer): GPT-2 tokenizer
        model (pytorch_pretrained_bert.modeling_gpt2.GPT2Model): GPT-2 Model
        bpe2word (str): how to turn the BPE vectors into word vectors.
            'last': last hidden state; 'avg': average hidden state.
        prevs_align (int): aligning location for the last word in the sequence.
            If provided, monotonicity is required.
        normalized (bool): whether to use normalized dot product (cosine similarity) for token similarity calculation
    
    Output:
        scores (torch.Tensor): unsorted one-token similarity scores
        indices (torch.LongTensor): matched indices in template_vec for each token
        states (list): corresponding GPT-2 hidden states
    '''
    assert bpe2word in ['last', 'avg']
    
    device = next(model.parameters()).device
    model.eval()
    
    if prevs_state is None:
        # beginning of sentence, the first token
        assert prevs_align is None, 'Nothing should be passed in when no history.'
    else:
        # in the middle of a sentence, sequential update
        assert prevs_state is not None, 'There should be history.'
 
    embeddings = []    # word embeddings
    states = []        # hidden states saved for sequential calculations
    with torch.no_grad():
        for bpe_encoding in bpe_encoding_grouped:
            # bpe_encoding is a tensor of bpe unit ids
            vec, past = model(bpe_encoding, past=prevs_state)
            # vec: size (n, len(bpe_encoding), 768)
            # past: a list of length 12, each of size (2, n, 12, len(bpe_encoding), 64)
            #       which records keys, values for 12 heads in each of the 12 layers
            # where n is the number of words of the same len(bpe_encoding) in the word list

            if bpe2word == 'last':
                embeddings.append(vec[:, -1, :])    # size (n, 768)
            elif bpe2word == 'avg':
                embeddings.append(vec.mean(dim=1))    # size (n, 768)
            else: # impossible
                raise ValueError
            
            past = torch.cat(past, dim=0)          # size (2 * 12, n, 12, len(bpe_encoding), 64)
            past = torch.split(past, 1, dim=1)    # list of length n, each of size (2 * 12, 1, 12, len(bpe_encoding), 64)
            states += past
    
    embeddings = torch.cat(embeddings, dim=0)       # size (#word_list, 768)
    states = [torch.chunk(s, 12, dim=0) for s in states]
    
    scores = []
    indices = []
    print('Calculating similarities ---')
#     start = time.time()
    scores, indices = OneTokenMatch(template_vec, embeddings, normalized=normalized, starting_loc=prevs_align)
#     print('Similarities: ' + timeSince(start))
        
    return scores, indices, states
"""
