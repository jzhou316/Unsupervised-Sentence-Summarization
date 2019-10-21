import torch


def prob_next(LMModel, vocab, text, hn=None, subvocab=None, clustermask=None, onscore=False, renorm=False):
    """
    Output the probability distribution for the next word based on a pretrained LM, given the previous text.
    If 'subvocab' is not None, the distribution is restricted on the specified sub-vocabulary.
    
    Input:
        LMModel: pretrained RNN LM model.
        vocab: full vocabulary. 'torchtext.vocab.Vocab'.
        text: previous words in the sentence.
        hn: initial hidden states to the LM.
        subvocab: sub-vocabulary. 'torch.LongTensor'.
        clustermask: a binary mask for each of the sub-vocabulary word. 'torch.ByteTensor' of size (len(sub-vocabulary), len(vocabulary)).
        onscore: whether to cluster on the raw scores before softmax layer, rather than cluster on the probabilities.
        renorm: whether to renormalize the probabilities over the sub-vocabulary. This parameter only works if 'onscore' is False.
        
    Output:
        (if subvocab is not None) subprobs: probability distribution over the sub-vocabulary.
        probs: probability distribution over the full vocabulary.
        hn: hidden states.
    """
    
    if clustermask is not None:
        assert subvocab is not None, 'clustermask provided but No subvocab provided.'
        
    if isinstance(text, str):
        text = text.split()
    
    textid = next(LMModel.parameters()).new_tensor([vocab.stoi[w] for w in text],
                 dtype=torch.long)
    with torch.no_grad():
        LMModel.eval()
        batch_text = textid.unsqueeze(1)
        embed = LMModel.embedding(batch_text)
        output, hn = LMModel.lstm(embed, hn)
        output = LMModel.proj(output)      # size: (seq_len, batch_size=1, vocab_size)
        
    probs = torch.nn.functional.softmax(output[-1].squeeze(), dim=0)
    
    if subvocab is None:
        # if no subvocab is provided, return the full probability distribution and hidden states
        return probs, hn
    
    ## cluster on the raw scores (rather than the probabilities) before passing to the softmax layer
    if onscore:
        scores = output[-1].squeeze()
        subscores = scores[subvocab]
        if clustermask is None:
            subprobs = torch.nn.functional.softmax(subscores, dim=0)
            return subprobs, probs, hn
        for i in range(len(subvocab)):
            subscores[i] = scores[clustermask[i]].sum()
        subprobs = torch.nn.functional.softmax(subscores, dim=0)
        return subprobs, probs, hn
    
    ## cluster on the probabilities
    subprobs = probs[subvocab]
    if clustermask is None:
        if renorm:
            subprobs = subprobs / subprobs.sum()
#             subprobs = torch.nn.functional.softmax(subprobs, dim=0)        # this makes the ratio p1/p2 between p1 and p2 (p1 > p2) smaller
        return subprobs, probs, hn
    
    for i in range(len(subvocab)):
        subprobs[i] = probs[clustermask[i]].sum()
    if renorm:
        subprobs = subprobs / subprobs.sum()
#         subprobs = torch.nn.functional.softmax(subprobs, dim=0)        # this makes the ratio p1/p2 between p1 and p2 (p1 > p2) smaller
    return subprobs, probs, hn


def prob_next_1step(LMModel, batch_text, hn=None, subvocab=None, clustermask=None, onscore=False, renorm=False, temperature=1):
    """
    Output the probability distribution for the next word based on a pretrained LM, carried in only one step of the forward pass.
    If 'subvocab' is not None, the distribution is restricted on the specified sub-vocabulary.
    This function is specifically used in the beam search.
    
    Input:
        LMModel: pretrained RNN LM model.
        batch_text: text id input to the language model, of size (seq_len=1, batch_size=onbeam_size).
        hn: hidden states to the LM, a tuple and each of size (num_layers * num_directions, batch_size=onbeam_size, hidden_size).
        subvocab: sub-vocabulary. 'torch.LongTensor'.
        clustermask: a binary mask for each of the sub-vocabulary word. 'torch.ByteTensor' of size (len(sub-vocabulary), len(vocabulary)).
        onscore: whether to cluster on the raw scores before softmax layer, rather than cluster on the probabilities.
        renorm: whether to renormalize the probabilities over the sub-vocabulary. This parameter only works if 'onscore' is False.
    
    Output:
        subprobs: probability distribution over the sub-vocabulary. Size: (batch_size=onbeam_size, subvocab_size)
        probs: probability distribution over the full vocabulary. Size: (batch_size=onbeam_size, vocab_size)
        hn: hidden states. Tuple, each of size (num_layers * num_directions, batch_size=onbeam_size, hidden_size).
    """
    
    if clustermask is not None:
        assert subvocab is not None, 'clustermask provided but No subvocab provided.'
        
    with torch.no_grad():
        LMModel.eval()
        embed = LMModel.embedding(batch_text)
        output, hn = LMModel.lstm(embed, hn)
        output = LMModel.proj(output)      # size: (seq_len=1, batch_size=onbeam_size, vocab_size)
    
    output = output / temperature
    probs = torch.nn.functional.softmax(output.squeeze(0), dim=1)    # size: (batch_size=onbeam_size, vocab_size)
       
    if subvocab is None:
        # if no subvocab is provided, return the full probability distribution and hidden states
        return probs, probs, hn
    
#     ## cluster on the raw scores (rather than the probabilities) before passing to the softmax layer
#     if onscore:
        
#         return 
    
    ## cluster on the probabilities
    subprobs = probs[:, subvocab]             # size: (batch_size=onbeam_size, subvocab_size)
    if clustermask is None:
        if renorm:
            subprobs = subprobs / torch.sum(subprobs, dim=1, keepdim=True)
#             subprobs = torch.nn.functional.softmax(subprobs, dim=1)        # this makes the ratio p1/p2 between p1 and p2 (p1 > p2) smaller
        return subprobs, probs, hn
    
    for i in range(len(subvocab)):
        subprobs[:, i] = probs[:, clustermask[i]].sum(dim=1)
    
    if renorm:
        subprobs = subprobs / torch.sum(subprobs, dim=1, keepdim=True)
#         subprobs = torch.nn.functional.softmax(subprobs, dim=1)        # this makes the ratio p1/p2 between p1 and p2 (p1 > p2) smaller
    return subprobs, probs, hn
    
    
def prob_sent(LMModel, vocab, text, hn=None, subvocab=None, clustermask=None, onscore=False, renorm=False, size_average=False):
    """
    Output the log-likelihood of a sentence based on a pretrained LM.
    If 'subvocab' is not None, the distribution is restricted on the specified sub-vocabulary.
    
    Input:
        LMModel: pretrained RNN LM model.
        vocab: full vocabulary. 'torchtext.vocab.Vocab'.
        text: previous words in the sentence.
        hn: initial hidden states to the LM.
        subvocab: sub-vocabulary. 'torch.LongTensor'.
        clustermask: a binary mask for each of the sub-vocabulary word. 'torch.ByteTensor' of size (len(sub-vocabulary), len(vocabulary)).
        onscore: whether to cluster on the raw scores before softmax layer, rather than cluster on the probabilities.
        renorm: whether to renormalize the probabilities over the sub-vocabulary. This parameter only works if 'onscore' is False.
        size_average: whether to average the log-likelihood according to the sequence length.
        
    Output:
        ll: log-likelihood of the given sentence evaluated by the pretrained LM.
        hn: hidden states.
    """
    
    if clustermask is not None:
        assert subvocab is not None, 'clustermask provided but No subvocab provided.'
        
    if isinstance(text, str):
        text = text.split()
        
    ## no subvocab is provided, operating on the full vocabulary
    textid = next(LMModel.parameters()).new_tensor([vocab.stoi[w] for w in text],
                 dtype=torch.long)
    if subvocab is None:
        with torch.no_grad():
            LMModel.eval()
            batch_text = textid.unsqueeze(1)
            embed = LMModel.embedding(batch_text)
            output, hn = LMModel.lstm(embed, hn)
            output = LMModel.proj(output)      # size: (seq_len, batch_size=1, vocab_size)
        ll = torch.nn.functional.cross_entropy(output.squeeze()[:-1, :], textid[1:], size_average=size_average, ignore_index=LMModel.padid)
        ll = -ll.item()
        return ll, hn
    
    ## subvocab is provided
    textid_sub = next(LMModel.parameters()).new_tensor([subvocab.numpy().tolist().index(vocab.stoi[w]) for w in text],
                                                       dtype=torch.long)
    subprobs_sent = torch.zeros(len(text) - 1, len(subvocab), device=next(LMModel.parameters()).device)
    for i in range(len(text) - 1):
        subprobs, probs, hn = prob_next(LMModel, vocab, text[i], hn, subvocab, clustermask, onscore, renorm)
        subprobs_sent[i] = subprobs
    ll = torch.nn.functional.nll_loss(torch.log(subprobs_sent), textid_sub[1:], size_average=size_average, ignore_index=LMModel.padid)
    ll = -ll.item()
    return ll, hn


def clmk_nn(embedmatrix, subvocab, normalized=True):
    """
    Generate 'clustermask', based on nearest neighbors, i.e. each word outside of the sub-vocabulary is assigned 
    to the group of its closest one in the sub-vocabulary.
    
    Input:
        embedmatrix: word embedding matrix. Default should be the output embedding from the RNN language model.
        subvocab: sub-vocabulary. 'torch.LongTensor'.
        normalized: whether to use the normalized dot product as the distance measure, i.e. cosine similarity.
    
    Output:
        clustermask: a binary mask for each of the sub-vocabulary word. 'torch.ByteTensor' of size (len(sub-vocabulary), len(vocabulary)).
    """
    
    submatrix = embedmatrix[subvocab]
    sim_table = torch.mm(submatrix, embedmatrix.t())
    if normalized:
        sim_table = sim_table / torch.ger(submatrix.norm(2, 1), embedmatrix.norm(2, 1))
    maxsim, maxsim_ind = torch.max(sim_table, dim=0)
    
    groups = []
    vocab_ind = torch.arange(len(embedmatrix), device=embedmatrix.device)
    clustermask = torch.zeros_like(sim_table, dtype=torch.uint8, device='cpu')
    for i in range(len(subvocab)):
        groups.append(vocab_ind[maxsim_ind == i].long())
        clustermask[i][groups[i]] = 1
    
    return clustermask


def clmk_cn(embedmatrix, subvocab, simthre=0.6, normalized=True):
    """
    Generate 'clustermask', based on the cone method, i.e. each word in the sub-vocabulary is joined by the closest words in a cone,
    specified by a cosine similarity threshold.
    
    Input:
        embedmatrix: word embedding matrix. Default should be the output embedding from the RNN language model.
        subvocab: sub-vocabulary. 'torch.LongTensor'.
        simthre: cosine similarity threshold.
        normalized: whether to use the normalized dot product as the distance measure, i.e. cosine similarity.
   
    Output:
        clustermask: a binary mask for each of the sub-vocabulary word. 'torch.ByteTensor' of size (len(sub-vocabulary), len(vocabulary)).
    """
    
    submatrix = embedmatrix[subvocab]
    sim_table = torch.mm(submatrix, embedmatrix.t())
    if normalized:
        sim_table = sim_table / torch.ger(submatrix.norm(2, 1), embedmatrix.norm(2, 1))
    
    clustermask = (sim_table > simthre).to('cpu')
    ## remove the indices that are already in the sub-vocabulary
    subvocabmask = torch.zeros_like(clustermask, dtype=torch.uint8)
    subvocabmask[:, subvocab] = 1
    clustermask = (clustermask ^ subvocabmask) & clustermask        # set difference
    for i in range(len(subvocab)):
        clustermask[i][subvocab[i]] = 1                     # add back the current word in the sub-vocabulary
    
    return clustermask
