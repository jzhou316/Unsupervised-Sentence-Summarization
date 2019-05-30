'''
Sequential calculation of word similarities in an embedding space.
Previous alignments are fixed once found.

Each time, do the best word vector matching between a template sentence and a list of single tokens, based on cosine similarities or dot products.
In the simplest case, do not require monotonicity.
'''

import torch


def OneTokenMatch(src, token_list, normalized=False, starting_loc=None):
    """
    Input:
        src: source sequence, such as a long sentence vector to be summarized.
        token_list: a list of word vectors to be matched with 'src'.
        starting_loc: aligning location for the last word in the sequence.
                      If provided, monotonicity is required.
    Output:
        similarities: the best similarity scores for each token in 'token_list'.
        indices: the matched indices in 'src' for the best scores for each token.
    """
    if isinstance(token_list, list):
        assert isinstance(token_list[0], torch.Tensor) and isinstance(src, torch.Tensor), 'source/template sequence must be torch.Tensor.'
        assert len(token_list[0].size()) == len(src.size()) == 2, 'input sequences must be 2D series.'
    elif isinstance(token_list, torch.Tensor):
        assert isinstance(src, torch.Tensor), 'source/template sequence must be torch.Tensor.'
        assert len(token_list.size()) == len(src.size()) == 2, 'input sequences must be 2D series.'
    else:
        raise TypeError
    
    if starting_loc is not None:
        # require monotonicity, by only looking at 'src' from or after 'starting_loc'
        # strict monotonicity
        assert starting_loc < len(src) - 1, 'last word already matched to the last token in template, when requiring strict monotonicity.'
        src = src[(starting_loc + 1):]
        # weak monotonicity
#         assert starting_loc < len(src)
#         src = src[starting_loc:]
    
    if isinstance(token_list, list):
        token_matrix = torch.cat(token_list, dim=0)
    elif isinstance(token_list, torch.Tensor):
        token_matrix = token_list
    else:
        raise TypeError
    sim_table = torch.mm(src, token_matrix.t())        # size: (src_len, token_list_len) or (truncated_src_len, token_list_len)
    
    if normalized:
        sim_table = sim_table / torch.ger(src.norm(2, 1), token_matrix.norm(2, 1))
        
    similarities, indices = torch.max(sim_table, dim=0)
    
    if starting_loc is not None:
        indices += starting_loc + 1        # strict monotonicity
#         indices += starting_loc            # weak monotonicity 
    
    return similarities, indices


def TokenMatch(src, tgt, mono=True, weakmono=False, normalized=True):
    """
    Calculate the similarity between two sentences by word embedding match and single token alignment.
    
    Input:
        src: source sequence word embeddings. 'torch.Tensor' of size (src_seq_len, embed_dim).
        tgt: short target sequence word embeddings to be matched to 'src'. 'torch.Tensor' of size (tgt_seq_len, embed_dim).
        mono: whether to constrain the alignments to be monotonic. Default: True.
        weakmono: whether to relax the alignment monotonicity to be weak (non-strict). Only effective when 'mono' is True. Default: False.
        normalized: whether to normalize the dot product in calculating word similarities, i.e. whether to use cosine similarity or just dot                     product. Default: True.
    
    Output:
        similarity: sequence similarity, by summing the max similarities of the best alignment.
        indices: locations in the 'src' sequence that each 'tgt' token is aligned to.
    """
    
    assert isinstance(src, torch.Tensor) and isinstance(tgt, torch.Tensor), 'input sequences must be torch.Tensor.'
    assert len(src.size()) == len(tgt.size()) == 2, 'input sequences must be 2D series.'
    
    sim_table = torch.mm(src, tgt.t())
    if normalized:
        sim_table = sim_table / torch.ger(src.norm(2, 1), tgt.norm(2, 1))
    
    if mono:
        src_len, tgt_len = sim_table.size()
        max_sim = []
        if weakmono:
            indices = [0]
            for i in range(1, tgt_len + 1):
                mi, ii = torch.max(sim_table[indices[i - 1]:, i - 1].unsqueeze(1), dim=0)
                max_sim.append(mi)
                indices.append(ii + indices[i - 1])
        else:
            indices = [-1]
            for i in range(1, tgt_len + 1):
                if indices[i - 1] == src_len - 1:
                    max_sim.append(sim_table[-1, i - 1].unsqueeze(0))
                    indices.append(indices[i - 1])
                else:
                    mi, ii = torch.max(sim_table[(indices[i - 1] + 1):, i - 1].unsqueeze(1), dim=0)
                    max_sim.append(mi)
                    indices.append(ii + indices[i - 1] + 1)
        max_sim = torch.cat(max_sim)
        indices = torch.cat(indices[1:])
    else:
        max_sim, indices = torch.max(sim_table, dim=0)
    
    similarity = torch.sum(max_sim)
    
    return similarity, indices
