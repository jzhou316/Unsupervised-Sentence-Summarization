import torch
import pickle
import time
import math
import sys
import os
from tqdm import tqdm
import argparse

from pre_closetables import ELMoBotEmbedding, findclosewords_vocab

from elmo_sequential_embedder import ElmoEmbedderForward
# from pre_word_list import findwordlist, findwordlist_screened
from pre_word_list import findwordlist_screened2
from lm_subvocab import clmk_nn
from beam_search import Beam
from utils import timeSince


def gensummary_elmo(template_vec,
                    ee,
                    vocab,
                    LMModel,
                    word_list,
                    subvocab,
                    clustermask=None,
                    mono=True,
                    renorm=True,
                    temperature=1,
                    elmo_layer='avg',
                    max_step = 20,
                    beam_width = 10,
                    beam_width_start = 10,
                    alpha=0.1,
                    alpha_start=0.1,
                    begineos=True,
                    stopbyLMeos=False,
                    devid=0,
                    **kwargs):
    """
    Unsupervised sentence summary generation using beam search, by contextual matching and a summary style language model.
    The contextual matching here is on top of pretrained ELMo embeddings.
    
    Input:
        template_vec: forward only ELMo embeddings of the source sentence. 'torch.Tensor' of size (3, seq_len, 512).
        ee: 'elmo_sequential_embedder.ElmoEmbedderForward' object.
        vocab: 'torchtext.vocab.Vocab' object. Should be the same as is used for the pretrained language model.
        LMModel: a pretrained language model on the summary sentences.
        word_list: a list of words in the vocabulary to work with. 'List'.
        subvocab: 'torch.LongTensor' consisting of the indices of the words corresponding to 'word_list'.
        clustermask: a binary mask for each of the sub-vocabulary word. 'torch.ByteTensor' of size (len(sub-vocabulary), len(vocabulary)). Default:None.
        mono: whether to keep monotonicity contraint. Default: True.
        renorm: whether to renormalize the probabilities over the sub-vocabulary. Default: True.
        Temperature: temperature applied to the softmax in the language model. Default: 1.
        elmo_layer: which ELMo layer to use as the word type representation. Choose from ['avg', 'cat', 'bot', 'mid', 'top']. Default: 'avg'.
        max_step: maximum number of beam steps.
        beam_width: beam width.
        beam_width_start: beam width of the first step.
        alpha: the amount of language model part used for scoring. The score is: (1 - \alpha) * similarity_logscore + \alpha * LM_logscore.
        begineos: whether to begin with the special '<eos>' token as is trained in the language model. Note that ELMo has its own special beginning token. Default: True.
        stopbyLMeos: whether to stop a sentence solely by the language model predicting '<eos>' as the top possibility. Default: False.
        devid: device id to run the algorithm and LSTM language models. 'int', default: 0. -1 for cpu.
        **kwargs: other arguments input to function <Beam.beamstep>. 
            E.g. normalized: whether to normalize the dot product when calculating the similarity, which makes it cosine similarity. Default: True.
                 ifadditive: whether to use an additive model on mixing the probability scores. Default: False.
    
    Output:
        beam: 'Beam' object, recording all the generated sequences.
        
    """
    device = 'cpu' if devid == -1 else f'cuda:{devid}'
    
    # Beam Search: initialization
    if begineos:
        beam = Beam(1, vocab, init_ids=[vocab.stoi['<eos>']], device=device,
                sim_score=0, lm_score=0, lm_state=None, elmo_state=None, align_loc=None)
    else:
        beam = Beam(1, vocab, init_ids=[None], device=device,
                sim_score=0, lm_score=0, lm_state=None, elmo_state=None, align_loc=None)
    
    # first step: start with 'beam_width_start' best matched words
    beam.beamstep(beam_width_start,
                  beam.combscoreK,
                  template_vec=template_vec,
                  ee=ee,
                  LMModel=LMModel,
                  word_list=word_list,
                  subvocab=subvocab,
                  clustermask=clustermask, 
                  alpha=alpha_start,
                  renorm=renorm,
                  temperature=temperature,
                  elmo_layer=elmo_layer,
                  normalized=True,
                  ifadditive=False,
                  **kwargs)
    
    # run beam search, until all sentences hit <EOS> or max_step reached
    for s in range(max_step):
        print(f'beam step {s+1} ' + '-' * 50 + '\n')
        beam.beamstep(beam_width,
                      beam.combscoreK,
                      template_vec=template_vec,
                      ee=ee,
                      LMModel=LMModel,
                      word_list=word_list,
                      subvocab=subvocab,
                      clustermask=clustermask,
                      mono=mono,
                      alpha=alpha,
                      renorm=renorm,
                      temperature=temperature,
                      stopbyLMeos=stopbyLMeos,
                      elmo_layer=elmo_layer,
                      normalized=True,
                      ifadditive=False,
                      **kwargs)
        # all beams reach termination
        if beam.endall:
            break
    
    return beam


def sortsummary(beam, beta=0):
    """
    Sort the generated summaries by beam search, with length penalty considered.
    
    Input:
        beam: 'Beam' object finished with beam search.
        beta: length penalty when sorting. Default: 0 (no length penalty).
    Output:
        ssa: 'List[Tuple]' of (score_avg, sentence, alignment, sim_score, lm_score).
    """
    sents = []
    aligns = []
    score_avgs = []
    sim_scores = []
    lm_scores = []
        
    for ks in beam.endbus:
        sent, rebeam = beam.retrieve(ks[0] + 1, ks[1])
        score_avg = ks[2] / (ks[1] ** beta)
                
        sents.append(sent)
        aligns.append(beam.retrieve_align(rebeam))
        score_avgs.append(score_avg)
        sim_scores.append(ks[3])
        lm_scores.append(ks[4])
        
    ssa = sorted([(score_avgs[i], sents[i], aligns[i], sim_scores[i], lm_scores[i]) for i in range(len(sents))], reverse=True)
    
    return ssa


def fixlensummary(beam, length=-1):
    """
    Pull out fixed length summaries from the beam search.
    
    Input:
        beam: 'Beam' object finished with beam search.
        length: wanted length of the summary.
    Output:
        ssa: 'List[Tuple]' of sorted (score, sentence, alignments, sim_score, lm_score).
    """
    assert length >=1 and length <= beam.step, 'invalid sentence length.'
    
    ssa = []
    for i in range(beam.K[length]):
        sent, rebeam = beam.retrieve(i + 1, l)
        ssa.append((beam.beamseq[length][i].score, sent, beam.retrieve_align(rebeam), beam.beamseq[length][i].sim_score, beam.beamseq[length][i].lm_score))
            
    return ssa


##### input arguments
arttxtpath = '../LM/data/Giga-sum/input_unk_250.txt'
#arttxtpath = '../LM/data/Giga-sum/input_unk_251-500.txt'
#arttxtpath = '../LM/data/Giga-sum/input_unk_501-750.txt'
#arttxtpath = '../LM/data/Giga-sum/input_unk_751-1000.txt'
#arttxtpath = '../LM/data/Giga-sum/input_unk_1001-1250.txt'
#arttxtpath = '../LM/data/Giga-sum/input_unk_1251-1500.txt'
#arttxtpath = '../LM/data/Giga-sum/input_unk_1501-1750.txt'
#arttxtpath = '../LM/data/Giga-sum/input_unk_1751-1951.txt'

#arttxtpath = '../LM/data/Giga-sum/input_unk.txt'

devid = 0

arttxtpath = '/n/rush_lab/users/jzhou/sentence-compression/dataclean/eval_src_1000_unk.txt'

vocab_path = '../LM/LSTM/models_sc/vocabsctgt.pkl'
modelclass_path = '../LM/LSTM'
model_path = '../LM/LSTM/models_sc/sctgt_LSTM_1024_untied.pth'
closeword = './voctbls/vocabsctgtCloseWord'
closeword_lmemb = './voctbls/vocabsctgtCloseWord_1024_untied_'
savedir = './results_sc_1024_untied/'


'''
arttxtpath = '/n/rush_lab/users/jzhou/sentence-compression/dataclean/eval_src_1000_unk.txt'

vocab_path = '../LM/LSTM/models_sc/vocabsctgt.pkl'
modelclass_path = '../LM/LSTM'
model_path = '../LM/LSTM/models_sc/sctgt_LSTM_untied.pth'
closeword = 'vocabsctgtCloseWord'
closeword_lmemb = 'vocabsctgtCloseWord_untied_'
savedir = './results_sc_untied/'

'''
'''
vocab_path = '../LM/LSTM/models/vocabTle.pkl'
modelclass_path = '../LM/LSTM'
model_path = '../LM/LSTM/models/Tle_LSTM_untied.pth'
closeword = './voctbls/vocabTleCloseWord'
closeword_lmemb = './voctbls/vocabTleCloseWord_untied_'
savedir = './results_LMuntied_elmo/'
'''

# vocab_path = '../LM/LSTM/models/vocabTle.pkl'
# modelclass_path = '../LM/LSTM'
# model_path = '../LM/LSTM/models/Tle_LSTM.pth'
# closeword = 'vocabTleCloseWord'
# closeword_lmemb = 'vocabTleCloseWord'
# savedir = './results/'


# vocab_path = '../LM/LSTM_LUC/models/vocabTle50k.pkl'
# modelclass_path = '../LM/LSTM_LUC'
# model_path = '../LM/LSTM_LUC/models/TleLUC_wtI_noB_0-0.0001-1Penalty.pth'
# closeword = 'vocabTle50kCloseWord'
# closeword_lmemb = 'vocabTle50kCloseWord'

# vocab_path = '../4.0_cluster/vocabTle.pkl'                   # vocabulary for the pretrained language model
# modelclass_path = '../LM/LSTM_MoS'
# model_path = '../LM/LSTM_MoS/models/LMModelMoSTle2.pth'    
# closeword = '../4.0_cluster/vocabTleCloseWord'        # character level word embeddings
# closeword_lmemb = 'vocabTleCloseWord'


##### beam search parameters
begineos = True
appendsenteos = True
eosavgemb = False
max_step = 20
beam_width = 10
beam_width_start = 10
# mono = True
renorm = False
cluster = True
temperature = 1
elmo_layer = 'avg'
alpha = 0.1
alpha_start = alpha
stopbyLMeos = False
# ifadditive = False
beta = 0.0

# find word list
numwords = 6
numwords_outembed = -1
numwords_freq = 500

# if fix generation length
fixedlen = False
genlen = '9'
# genlen = [9]               # including the starting '<eos>' token, and can include the ending '<eos>' token as well (if not 'stobbyLMeos')
    

def parse_args():
    parser = argparse.ArgumentParser(description='Unsupervisely generate summaries from source file.')
    # source file
    parser.add_argument('--src', type=str, default=arttxtpath, help='source sentences file')
    parser.add_argument('--devid', type=int, default=devid, help='device id; -1 for cpu')
    # preparations
    parser.add_argument('--vocab', type=str, default=vocab_path, help='vocabulary file')
    parser.add_argument('--modelclass', type=str, default=modelclass_path, help='location of the model class definition file')
    parser.add_argument('--model', type=str, default=model_path, help='pre-trained language model')
    parser.add_argument('--closeword', type=str, default=closeword, help='character embedding close word tables')
    parser.add_argument('--closeword_lmemb', type=str, default=closeword_lmemb, help='LM output embedding close word tables')
    parser.add_argument('--savedir', type=str, default=savedir, help='directory to save results')
    # beam search parameters
    parser.add_argument('--begineos', type=int, default=int(begineos), help='whether to start with <eos>')
    parser.add_argument('--appendsenteos', type=int, default=int(appendsenteos), help='whether to append <eos> at the end of source sentence')
    parser.add_argument('--eosavgemb', type=int, default=int(eosavgemb), help='whether to encode <eos> using average hidden states')
    parser.add_argument('--max_step', type=int, default=max_step, help='maximum beam step')
    parser.add_argument('--beam_width', type=int, default=beam_width, help='beam width')
    parser.add_argument('--beam_width_start', type=int, default=beam_width_start, help='beam width at first step')
    parser.add_argument('--renorm', type=int, default=int(renorm), help='whether to renormalize the probabilities over the sub-vocabulary')
    parser.add_argument('--cluster', type=int, default=int(cluster), help='whether to do clustering for the sub-vocabulary probabilities')
    parser.add_argument('--temp', type=float, default=temperature, help='temperature used to smooth the output of the softmax layer')
    parser.add_argument('--elmo_layer', type=str, default=elmo_layer, choices=['bot', 'mid', 'top', 'avg', 'cat'], help='elmo layer to use')
    parser.add_argument('--alpha', type=float, default=alpha, help='mixture coefficient for LM')
    parser.add_argument('--alpha_start', type=float, default=alpha_start, help='mixture coefficient for LM for the first step')
    parser.add_argument('--stopbyLMeos', type=int, default=int(stopbyLMeos), help='whether to stop the sentence solely by LM <eos> prediction')
    parser.add_argument('--beta', type=int, default=beta, help='length penalty')
    parser.add_argument('--n', type=int, default=numwords, help='number of closest words for each token to form the candidate list')
    parser.add_argument('--ns', type=int, default=numwords_outembed, help='number of closest words for each token in the output embedding for each token to screen the candidate list')
    parser.add_argument('--nf', type=int, default=numwords_freq, help='number of the most frequent words in the vocabulary to keep in the candidate list')
    parser.add_argument('--fixedlen', type=int, default=int(fixedlen), help='whether to generate fixed length summaries')
    parser.add_argument('--genlen', type=str, default=genlen, help='lengths of summaries to be generated; should be comma separated')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    ##### input arguments
    arttxtpath = args.src

    devid = args.devid
    
    vocab_path = args.vocab                   # vocabulary for the pretrained language model
    modelclass_path = args.modelclass
    model_path = args.model
    
    closewordsim_path = args.closeword + 'Sims.pkl'
    closewordind_path = args.closeword + 'Indices.pkl'        # character level word embeddings
    closewordsim_outembed_path = args.closeword_lmemb + 'Sims_outembed_' + os.path.splitext(os.path.basename(model_path))[0] + '.pkl'
    closewordind_outembed_path = args.closeword_lmemb + 'Indices_outembed_' + os.path.splitext(os.path.basename(model_path))[0] + '.pkl'    
    
    device = 'cpu' if devid == -1 else f'cuda:{devid}'

    ##### beam search parameters
    begineos = args.begineos
    appendsenteos = args.appendsenteos
    eosavgemb = args.eosavgemb if appendsenteos else False
    max_step = args.max_step
    beam_width = args.beam_width
    beam_width_start = args.beam_width_start
    mono = True
    renorm = args.renorm
    cluster = args.cluster
    temp = args.temp
    elmo_layer = args.elmo_layer
    alpha = args.alpha
    alpha_start = args.alpha_start
    stopbyLMeos = args.stopbyLMeos
    ifadditive = False
    beta = args.beta
    numwords = args.n
    numwords_outembed = args.ns if args.ns != -1 else numwords
    numwords_freq = args.nf
    fixedlen = args.fixedlen
    genlen = list(map(int, args.genlen.split(',')))        # including the starting '<eos>' token
                                            # and can include the ending '<eos>' token as well (if not 'stobbyLMeos')
    
    ##### read in the article/source sentences to be summarized
    g = open(arttxtpath, 'r')
    sents = [line.strip() for line in g if line.strip()]        
    g.close()
    nsents = len(sents)
    
    ##### load the ELMo forward embedder class
    ee = ElmoEmbedderForward(cuda_device=devid)
    
    ##### load vocabulary and the pre-trained language model
    vocab = pickle.load(open(vocab_path, 'rb'))
    
    if modelclass_path not in sys.path:
        sys.path.insert(1, modelclass_path)  # this is for torch.load to load the entire model; the model class file must be included in the search path
    LMModel = torch.load(model_path, map_location=torch.device(device))
    embedmatrix = LMModel.proj.weight
    
    ##### check if the close_tables exist already; if not, generate
    if not os.path.exists(closewordind_path):
        # character embeddings of the vocabulary
        embedmatrix_cnn = ELMoBotEmbedding(vocab.itos, device=devid)
        values_cnn, indices_cnn = findclosewords_vocab(vocab, embedmatrix_cnn, numwords=500)
        # save results
        pickle.dump(values_cnn, open(closewordsim_path, 'wb'))
        pickle.dump(indices_cnn, open(closewordind_path, 'wb'))
    
    if not os.path.exists(closewordind_outembed_path):
        values, indices = findclosewords_vocab(vocab, embedmatrix, numwords=500)
        # save results
        pickle.dump(values, open(closewordsim_outembed_path, 'wb'))
        pickle.dump(indices, open(closewordind_outembed_path, 'wb'))
    
    closewordind = pickle.load(open(closewordind_path, 'rb'))
    closewordind_outembed = pickle.load(open(closewordind_outembed_path, 'rb'))
    
    ##### generate save file name
    basename = os.path.basename(arttxtpath)
    basename = os.path.splitext(basename)[0]
    
    savedir = args.savedir
#     savedir = './results/'
    
    smrypath = os.path.join(savedir, 'smry_') + basename + f'_Ks{beam_width_start}' + f'_clust{int(cluster)}'
    
    if renorm:
        smrypath += f'_renorm{int(renorm)}'
    if temp != 1:
        smrypath += f'_temper{temp}'
    if elmo_layer != 'avg':
        smrypath += f'_EL{elmo_layer}'

    smrypath += f'_eosavg{int(eosavgemb)}' + f'_n{numwords}'
    
    if numwords_outembed != numwords:
        smrypath += f'_ns{numwords_outembed}'
    if numwords_freq != 500:
        smrypath += f'_nf{numwords_freq}'
    if beam_width != 10:
        smrypath += f'_K{beam_width}'
    if stopbyLMeos:
        smrypath += f'_soleLMeos'
############################
#    smrypath += '_close1'
############################    
    if alpha_start != alpha:
        smrypath += f'_as{alpha_start}'
    if fixedlen:
        genlen = sorted(genlen)
        smrypath_list = [smrypath + f'_length{l - 1}' + f'_a{alpha}' + '_all.txt' for l in genlen]
    else:
        smrypath += f'_a{alpha}' + f'_b{beta}' + '_all.txt'
    
    ##### run summary generation and write to file
    if fixedlen:
        g_list = [open(fname, 'w') for fname in smrypath_list]
    else:
        g = open(smrypath, 'w')
        
    start = time.time()
    for ind in tqdm(range(nsents)):
        template = sents[ind].strip('.').strip()        # remove '.' at the end
        if appendsenteos:
            template += ' <eos>'

        ### Find the close words to those in the template sentence
#        word_list, subvocab = findwordlist(template, closewordind, vocab, numwords=1, addeos=True)
#        word_list, subvocab = findwordlist_screened(template, closewordind, closewordind_outembed, vocab, numwords=6, addeos=True)
        word_list, subvocab = findwordlist_screened2(template, closewordind, closewordind_outembed, vocab, numwords=numwords, numwords_outembed=numwords_outembed, numwords_freq=numwords_freq, addeos=True)
        if cluster:
            clustermask = clmk_nn(embedmatrix, subvocab)

        ### ELMo embedding of the template sentence
        if eosavgemb is False:
            template_vec, _ = ee.embed_sentence(template.split(), add_bos=True)
        else:
            tt = template.split()[:-1]
            hiddens = []
            template_vec = None
            current_hidden = None
            for i in range(len(tt)):
                current_embed, current_hidden = ee.embed_sentence([tt[i]], add_bos=True if i == 0 else False, initial_state=current_hidden)
                hiddens.append(current_hidden)
                template_vec = current_embed if template_vec is None else torch.cat([template_vec, current_embed], dim=1) 
            hiddens_h, hiddens_c = zip(*hiddens)
            hiddens_avg = (sum(hiddens_h) / len(hiddens_h), sum(hiddens_c) / len(hiddens_c))
            eosavg, _ = ee.embed_sentence(['<eos>'], initial_state=hiddens_avg)
            template_vec = torch.cat([template_vec, eosavg], dim=1)

        ### beam search
        max_step_temp = min([len(template.split()), max_step])
        beam = gensummary_elmo(template_vec,
                               ee,
                               vocab,
                               LMModel,
                               word_list,
                               subvocab,
                               clustermask=clustermask if cluster else None,
                               renorm=renorm,
                               temperature=temp,
                               elmo_layer=elmo_layer,
                               max_step=max_step_temp,
                               beam_width=beam_width,
                               beam_width_start=beam_width_start,
                               mono=True, 
                               alpha=alpha,
                               alpha_start=alpha_start,
                               begineos=begineos,
                               stopbyLMeos=stopbyLMeos,
                               devid=devid)
        
        ### sort and write to file
        if fixedlen:
            for j in range(len(genlen) - 1, -1, -1):
                g_list[j].write('-' * 5 + f'<{ind+1}>' + '-' * 5 + '\n')
                g_list[j].write('\n')
                if genlen[j] <= beam.step:
                    ssa = fixlensummary(beam, length=genlen[j])
                    if ssa == []:
                        g_list[j].write('\n')
                    else:
                        for m in range(len(ssa)):
                            g_list[j].write(' '.join(ssa[m][1][1:]) + '\n')
                            g_list[j].write('{:.3f}'.format(ssa[m][0]) + '   ' + '{:.3f}'.format(ssa[m][3])
                                       + '   ' + '{:.3f}'.format(ssa[m][4]) + '\n')
                            g_list[j].writelines(['%d,   ' % loc for loc in ssa[m][2]])
                            g_list[j].write('\n')
                        g_list[j].write('\n')
                else:
                    g_list[j].write('\n')
        else:
            ssa = sortsummary(beam, beta=beta)
            g.write('-' * 5 + f'<{ind+1}>' + '-' * 5 + '\n')
            g.write('\n')
            if ssa == []:
                g.write('\n')
            else:
                for m in range(len(ssa)):
                    g.write(' '.join(ssa[m][1][1:]) + '\n')
                    g.write('{:.3f}'.format(ssa[m][0]) + '   ' + '{:.3f}'.format(ssa[m][3]) + '   ' + '{:.3f}'.format(ssa[m][4]) + '\n')
                    g.writelines(['%d,   ' % loc for loc in ssa[m][2]])
                    g.write('\n')
                g.write('\n')
                
    print('time elapsed %s' % timeSince(start))
    if fixedlen:
        for gg in g_list:
            gg.close()
        print('results saved to: %s' % (("\n" + " " * 18).join(smrypath_list)))        
    else:
        g.close()
        print(f'results saved to: {smrypath}')

