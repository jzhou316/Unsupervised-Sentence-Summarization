import torch
import pickle
import math
from tqdm import tqdm

import sys

from elmo_sequential_embedder import ElmoEmbedderForward
from sim_embed_score import pickElmoForwardLayer


def ELMoBotEmbedding(itos, device=-1):
    """
    itos: List[str]. A list of words consisting of the vocabulary.
    device: int. -1 for cpu.
    """
    ee = ElmoEmbedderForward(cuda_device=device)
    vocab_vecs, _ = zip(*ee.embed_sentences([[w] for w in itos], add_bos=True, batch_size=1024))
    vocab_vecs = [pickElmoForwardLayer(vec, 'bot') for vec in vocab_vecs]
    embedmatrix = torch.cat(vocab_vecs, dim=0)    # size: (vocab_size, embed_size)
    
    return embedmatrix


def findclosewords_vocab(vocab, embedmatrix, numwords=500, normalized=True, device='cpu'):
    """
    Find closest words for every word in the vocabulary.
    """
    v = len(vocab)
    assert v == len(embedmatrix)
    
    embedmatrix = embedmatrix.to(device)
    
    chunk_size = 1000        # to solve the problem of out of memory
    
    if v > chunk_size:
        n = math.ceil(v / chunk_size)
    else:
        n = 1
    values = None
    indices = None
    start = 0
    for i in tqdm(range(n)):
        embedmatrix_chunk = embedmatrix[start:(start + chunk_size), :]
        start = start + chunk_size
        
        sim_table = torch.mm(embedmatrix_chunk, embedmatrix.t())
        if normalized:
            sim_table = sim_table / torch.ger(embedmatrix_chunk.norm(2, 1), embedmatrix.norm(2, 1))
    
        values_chunk, indices_chunk = sim_table.topk(numwords, dim=1)        
        values = values_chunk if values is None else torch.cat([values, values_chunk], dim=0)
        indices = indices_chunk if indices is None else torch.cat([indices, indices_chunk], dim=0)
                                            
    return values.to('cpu'), indices.to('cpu')        # values and indices have size (vocab_len, numwords)
    

if __name__ == '__main__':
    
    vocab_path = '../4.0_cluster/vocabTle.pkl'                   # vocabulary for the pretrained language model
    closewordsim_path = '../4.0_cluster/vocabTleCloseWordSims.pkl'
    closewordind_path = '../4.0_cluster/vocabTleCloseWordIndices.pkl'        # character level word embeddings
    closewordsim_outembed_path = 'vocabTleCloseWordSims_outembed_MoS.pkl'
    closewordind_outembed_path = 'vocabTleCloseWordIndices_outembed_MoS.pkl'
    modelclass_path = '../LSTM_MoS'
    model_path = '../LSTM_MoS/models/LMModelMoSTle2.pth'
    
#     vocab_path = '../LSTM_LUC/vocabTle50k.pkl'                   # vocabulary for the pretrained language model
#     closewordsim_path = 'vocabTle50kCloseWordSims.pkl'
#     closewordind_path = 'vocabTle50kCloseWordIndices.pkl'        # character level word embeddings
#     closewordsim_outembed_path = 'vocabTle50kCloseWordSims_outembed_wtI.pkl'
#     closewordind_outembed_path = 'vocabTle50kCloseWordIndices_outembed_wtI.pkl'
#     modelclass_path = '../LSTM_LUC'
#     model_path = '../LSTM_LUC/models/TleLUC_wtI_0-0.0001-1Penalty.pth'
    
    # vocabulary
    vocab = pickle.load(open(vocab_path, 'rb'))
    
#     # character embeddings of the vocabulary
#     embedmatrix_cnn = ELMoBotEmbedding(vocab.itos, device=0)
#     values_cnn, indices_cnn = findclosewords_vocab(vocab, embedmatrix_cnn, numwords=500)
    
#     # save results
#     pickle.dump(values_cnn, open(closewordsim_path, 'wb'))
#     pickle.dump(indices_cnn, open(closewordind_path, 'wb'))
    
    # output embeddings of the vocabulary
    modelclass_path = modelclass_path
    if modelclass_path not in sys.path:
        sys.path.insert(1, modelclass_path)  # this is for torch.load to load the entire model; the model class file must be included in the search path
    LMModel = torch.load(model_path, map_location=torch.device('cpu'))
    embedmatrix = LMModel.proj_vocab.weight
    values, indices = findclosewords_vocab(vocab, embedmatrix, numwords=500)
    
    # save results
    pickle.dump(values, open(closewordsim_outembed_path, 'wb'))
    pickle.dump(indices, open(closewordind_outembed_path, 'wb'))

