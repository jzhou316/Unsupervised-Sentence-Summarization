import torch


def findwordlist(template, closewordind, vocab, numwords=10, addeos=False):
    """
    Based on a template sentence, find the candidate word list.
    
    Input:
        template: source sentence.
        closewordind: precalculated 100 closest word indices (using character embeddings). torch.LongTensor.
        vocab: full vocabulary.
        numwords: number of closest words per word in the template.
        addeos: whether to include '<eos>' in the candidate word list.
    """
    if isinstance(template, str):
        template = template.split()
    templateind = closewordind.new_tensor([vocab.stoi[w] for w in template])
    # subvocab = closewordind[templateind, :numwords].flatten().cpu()  # torch.flatten() only exists from PyTorch 0.4.1
    subvocab = closewordind[templateind, :numwords].view(-1).cpu()
    if addeos:
        subvocab = torch.cat([subvocab, torch.LongTensor([vocab.stoi['<eos>']])])
    subvocab = subvocab.unique(sorted=True)
    word_list = [vocab.itos[i] for i in subvocab]
    
    return word_list, subvocab


def findwordlist_screened(template, closewordind, closewordind_outembed, vocab, numwords=10, addeos=False):
    """
    Based on a template sentence, find the candidate word list, according to the character level RNN embeddings but
    screened by the output embeddings.
    
    Input:
        template: source sentence.
        closewordind: precalculated 100 closest word indices (using character embeddings). torch.LongTensor.
        closewordind_embed: same as 'closewordind', but using output embeddings.
        vocab: full vocabulary.
        numwords: number of closest words per word in the template.
        addeos: whether to include '<eos>' in the candidate word list.
    """
    if isinstance(template, str):
        template = template.split()
    templateind = closewordind.new_tensor([vocab.stoi[w] for w in template])
    
    subvocab = closewordind[templateind, :numwords].view(-1).cpu()
    subvocab_embed = closewordind_outembed[templateind, 1:numwords].view(-1).cpu()
    subvocab_intemplate = closewordind[templateind, 0].view(-1).cpu()
    
    subvocab_mask = torch.zeros(len(vocab), dtype=torch.uint8, device=subvocab.device)
    subvocab_mask[subvocab] = 1
    subvocab_embed_mask = torch.zeros(len(vocab), dtype=torch.uint8, device=subvocab.device)
    subvocab_embed_mask[subvocab_embed] = 1
    
    subvocab_screened_mask = (subvocab_mask ^ subvocab_embed_mask) & subvocab_mask
    subvocab_screened_mask[subvocab_intemplate] = 1        # add back the words in the template sentence
    if addeos:
        subvocab_screened_mask[vocab.stoi['<eos>']] = 1
    
    subvocab_screened = torch.arange(len(vocab), dtype=torch.long, device=subvocab.device)
    subvocab_screened = subvocab_screened[subvocab_screened_mask]
    
    word_list = [vocab.itos[i] for i in subvocab_screened]
    
    return word_list, subvocab_screened


def findwordlist_screened2(template, closewordind, closewordind_outembed, vocab, numwords=10,
                           numwords_outembed=None, numwords_freq=500, addeos=False):
    """
    Based on a template sentence, find the candidate word list, according to the character level RNN embeddings but
    screened by the output embeddings, and keep the words that are in the top 'numwords_freq' list in the vocabulary.
    
    Input:
        template: source sentence.
        closewordind: precalculated 100 closest word indices (using character embeddings). torch.LongTensor.
        closewordind_embed: same as 'closewordind', but using output embeddings.
        vocab: full vocabulary.
        numwords: number of closest words per word in the template.
        numwords_outembed: number of closest words per word in the output embedding to be screened out.
        numwords_freq: number of the most frequent words in the vocabulary to remain.
        addeos: whether to include '<eos>' in the candidate word list.
    """
    if numwords_outembed is None:
        numwords_outembed = numwords
        
    if numwords_outembed <= 1:
        return findwordlist(template, closewordind, vocab, numwords=numwords, addeos=addeos)
    
    if isinstance(template, str):
        template = template.split()
    templateind = closewordind.new_tensor([vocab.stoi[w] for w in template])
    
    subvocab = closewordind[templateind, :numwords].view(-1).cpu()
    subvocab_embed = closewordind_outembed[templateind, 1:numwords_outembed].view(-1).cpu()
    subvocab_intemplate = closewordind[templateind, 0].view(-1).cpu()
    
    subvocab_mask = torch.zeros(len(vocab), dtype=torch.uint8, device=subvocab.device)
    subvocab_mask[subvocab] = 1
    subvocab_embed_mask = torch.zeros(len(vocab), dtype=torch.uint8, device=subvocab.device)
    subvocab_embed_mask[subvocab_embed[subvocab_embed >= numwords_freq]] = 1    # never remove the most frequent words
    
    subvocab_screened_mask = (subvocab_mask ^ subvocab_embed_mask) & subvocab_mask
    subvocab_screened_mask[subvocab_intemplate] = 1        # add back the words in the template sentence
    if addeos:
        subvocab_screened_mask[vocab.stoi['<eos>']] = 1
    
    subvocab_screened = torch.arange(len(vocab), dtype=torch.long, device=subvocab.device)
    subvocab_screened = subvocab_screened[subvocab_screened_mask]
    
    word_list = [vocab.itos[i] for i in subvocab_screened]
    
    return word_list, subvocab_screened
