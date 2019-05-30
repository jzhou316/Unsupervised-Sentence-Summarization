import torch
import torch.nn.functional as F
from sim_embed_score import simScoreNext, simScoreNext_GPT2
# from lmsubvocab import prob_next
from lm_subvocab import prob_next_1step
from utils import timeSince
import nltk
import math
import time


lemma = nltk.wordnet.WordNetLemmatizer()

class BeamUnit():
    def __init__(self, word_id, pre_loc, cur_loc, score, seq_len, vocab, **kwargs):
        self.score = score
        self.word_id = word_id
        self.pre_loc = pre_loc
        self.cur_loc = cur_loc
        self.seq_len = seq_len
        self.vocab = vocab
        for k, v in kwargs.items():
            setattr(self, k, v)
            

class Beam():
    def __init__(self, init_K, vocab, init_ids, device=None, **kwargs):
        assert init_K >= 1 and init_K <= len(vocab), 'Initial beam size should be in [1, len(vocab)]!'
        assert init_K == len(init_ids), 'Initial beam size should equal to the length of initial ids!'
        self.K = [init_K]                # dynamic beam size
        self.vocab = vocab
        self.step = 0
        self.device = device
        self.endbus = []                 # ending BeamUnits
        self.endall = False              # if all beams reach the termination
        if init_ids == [None]:           # A special initial id
            seq_len = 0
        else:
            seq_len = 1
        self.beamseq = [[BeamUnit(word_id, pre_loc=None, cur_loc=i, score=0, seq_len=seq_len, vocab=vocab, **kwargs)
                         for (i, word_id) in enumerate(init_ids)]]
        # Note: for the reason of unifying different language models, all the beams start from one single unit: 
        #       for similarity LM, init_ids should always be [None];
        #       for normal LM, init_ids should be of your own pick (since the current LM was not trained with 
        #       a special BOS token, it must start with some given token).
        
    def beamstep(self, K, score_funcK, **kwargs):
        """
        K: beam size next step
        score_func: a function that takes in a list of BeamUnit and returns the next top K BeamUnit based on some scores
        """
        if self.endall:
            raise ValueError('Beam.endall flag is already raised. No need to do beamstep.')
        
        nexttopKK, endbus = score_funcK(self.beamseq[-1], K, **kwargs)
        self.endbus += endbus
                
        if nexttopKK == []:
            print('All beams reach EOS. Beamstep stops.')
            self.endall = True
        else:
            self.beamseq.append(nexttopKK)               # TO DO: add termination condition
            self.K.append(len(nexttopKK))
            self.step += 1
        
    def beamcut(self, K, score_func=None, **kwargs):
        """
        Cut the current beam width down to K (top K).
        """
        assert K > 0, 'Beam width K should be positive!'
        if K >= self.K[-1]:
            print('No need to cut.')
        else:
            if score_func is None:
                self.beamseq[-1] = self.beamseq[-1][0:K]
                self.K[-1] = K
            else:
                ll = [score_func(text=self.retrieve(k+1)[0], **kwargs) for k in range(self.K[-1])]
                ll_sorted = sorted(list(zip(range(len(ll)), ll)), key=lambda x: x[1], reverse=True)[0:K]
                ll_idx, _ = zip(*ll_sorted)
                self.beamseq[-1] = [self.beamseq[-1][i] for i in range(self.K[-1]) if i in ll_idx]
                assert len(self.beamseq[-1]) == K
                for k, bu in enumerate(self.beamseq[-1]):
                    bu.cur_loc = k
                self.K[-1] = K
                return ll, ll_sorted
    
    def beamselect(self, indices=[0]):
        """
        Select the beams (at last step) according to indices.
        Default: select the first beam, which is equivalent to self.beamcut(1).
        """
        indices = sorted(list(set(indices)))        # indices: no repeated numbers, and should be sorted
        assert indices[-1] < self.K[-1], 'Index out of range (beamwidth).'
        self.beamseq[-1] = [self.beamseq[-1][i] for i in indices]
        for k, bu in enumerate(self.beamseq[-1]):
            bu.cur_loc = k
        self.K[-1] = len(self.beamseq[-1])
        
    def beamback(self, seq_len):
        """
        Trace back the beam at seq_len.
        """
        assert seq_len <= len(self.beamseq), 'seq_len larger than maximum.'
        if self.beamseq[0][0].word_id is None:
            self.beamseq = self.beamseq[0:(seq_len + 1)]
            self.K = self.K[0:(seq_len + 1)]
            self.step = seq_len
        else:
            self.beamseq = self.beamseq[0:seq_len]
            self.K = self.K[0:seq_len]
            self.step = seq_len - 1
        self.endall = False
        
    def retrieve(self, k, seq_len=-1):
        """
        Retrieve the k-th ranked generated sentence.
        """
  
        if self.beamseq[0][0].word_id is not None and seq_len > 0:
            # for a normal LM
            seq_len -= 1
        
        assert k >= 1 and k <= self.K[seq_len], 'k must be in [1, the total number of beams at seq_len]!'
        
        rebeam = [self.beamseq[seq_len][k - 1]]
        n = seq_len
        while rebeam[0].pre_loc is not None:
            n -= 1
            rebeam = [self.beamseq[n][rebeam[0].pre_loc]] + rebeam
        sent = [self.vocab.itos[bu.word_id] for bu in rebeam if bu.word_id is not None]
        
        return sent, rebeam
    
    def retrieve_align(self, rebeam):
        """
        Should be run after calling Beam.retrieve(...).
        """
        align_locs = [bu.align_loc.item() for bu in rebeam if bu.word_id is not None and bu.align_loc is not None]
        return align_locs
    
    def retrieve_endbus(self):
        """
        Retrieve the complete sentences acquired by beam steps.
        """
        sents = []
        aligns = []
        score_avgs = []
        for ks in self.endbus:
            sent, rebeam = self.retrieve(ks[0] + 1, ks[1])
            score_avg = ks[2] / ks[1]
            
            sents.append(sent)
            aligns.append(self.retrieve_align(rebeam))
            score_avgs.append(score_avg)
        
        return sents, aligns, score_avgs
    
    def simscore(self, bu, K, template_vec, ee, word_list=None, mono=False,
                 batch_size=1024, normalized=True, elmo_layer='avg'):
        """
        Score function based on sentence similarities.
        """        
        if word_list is None:
            word_list = self.vocab.itos
        scores, indices, states = simScoreNext(template_vec, word_list, ee,
                                              prevs_state=bu.elmo_state, batch_size=batch_size,
                                              prevs_align=bu.align_loc if mono else None,
                                              normalized=normalized, elmo_layer=elmo_layer)
        scores_prob = torch.nn.functional.log_softmax(scores, dim=0)
        
        sorted_scores, sorting_indices = torch.sort(scores)
        
        nexttopK = [BeamUnit(self.vocab.stoi[word_list[i]], bu.cur_loc, None, scores_prob[i].item() + bu.score,
                             bu.seq_len + 1, self.vocab, elmo_state=states[i], align_loc=indices[i].item()) 
                    for i in sorting_indices[0:(K + 5)]
                    # do not allow repeated words consecutively
                    if lemma.lemmatize(self.vocab.itos[bu.word_id]) != lemma.lemmatize(word_list[i])]
        nexttopK = nexttopK[0:K]
        
        return nexttopK
     
    def lmscore(self, bulist, K, LMModel, word_list=None, subvocab=None, clustermask=None, renorm=False, temperature=1):
        """
        Score function based on a pretrained RNN language model.
        """
        # note that LMModel should have the same vocab as that in Beam()
        
        ## when no candidate word list is provided, use the full vocabulary
        if word_list is None:
            word_list = self.vocab.itos
            subvocab=None
            clustermask=None
        
        if self.device is not None:
            LMModel = LMModel.cuda(device=self.device)
        LMModel.eval()
        with torch.no_grad():
            onbeam_ids = list(range(len(bulist)))
            batch_text = next(LMModel.parameters()).new_tensor([bulist[i].word_id for i in onbeam_ids],
                                                               dtype=torch.long).unsqueeze(0)
            if bulist[onbeam_ids[0]].lm_state is None:
                # 'lm_state' for the current beam is either all 'None' or all not 'None'.
                batch_hn = None
            else:
                batch_hn = (torch.cat([bulist[i].lm_state[0] for i in onbeam_ids], dim=1),
                           torch.cat([bulist[i].lm_state[1] for i in onbeam_ids], dim=1))  
            subprobs, probs, hn = prob_next_1step(LMModel, batch_text, hn=batch_hn,
                                                  subvocab=subvocab, clustermask=clustermask, onscore=False, renorm=renorm,
                                                  temperature=temperature)
        # convert the hidden state tuple into a list of tuples, corresponding to each beam sequence
        hn = list(zip(torch.chunk(hn[0], chunks=len(onbeam_ids), dim=1), torch.chunk(hn[1], chunks=len(onbeam_ids), dim=1)))
        lm_cum_logprob = subprobs.new_tensor([bulist[i].lm_score for i in onbeam_ids]).unsqueeze(1) + torch.log(subprobs)
        lm_cum_logprob = lm_cum_logprob.view(-1)      # this is the cumulative log probabilities
        
        ## rank and update
        if K > len(lm_cum_logprob):
            scores_sorted, ids_sorted = lm_cum_logprob.sort(descending=True)
            nexttopKK = [BeamUnit(self.vocab.stoi[word_list[i % len(word_list)]],
                                 bulist[onbeam_ids[i // len(word_list)]].cur_loc,
                                 m,
                                 scores_sorted[m].item(),
                                 bulist[onbeam_ids[i // len(word_list)]].seq_len + 1,
                                 self.vocab,
                                 lm_score=lm_cum_logprob[i].item(),
                                 lm_state=hn[i // len(word_list)])
                         for (m, i) in enumerate(ids_sorted)]
        else:
            scores_topK, ids_topK = lm_cum_logprob.topk(K)
            nexttopKK = [BeamUnit(self.vocab.stoi[word_list[i % len(word_list)]],
                                 bulist[onbeam_ids[i // len(word_list)]].cur_loc,
                                 m,
                                 scores_topK[m].item(),
                                 bulist[onbeam_ids[i // len(word_list)]].seq_len + 1,
                                 self.vocab,
                                 lm_score=lm_cum_logprob[i].item(),
                                 lm_state=hn[i // len(word_list)])
                        for (m, i) in enumerate(ids_topK)]
        
        endbus = []
        
        return nexttopKK, endbus
    
    def combscoreK(self, bulist, K, template_vec, ee, LMModel,
                  word_list=None, subvocab=None, clustermask=None,
                  mono=True, batch_size=1024, normalized=True, renorm=False, temperature=1,
                  elmo_layer='avg', alpha=0.01, stopbyLMeos=False, ifadditive=False):
        """
        Given a list of 'BeamUnit', score the next tokens from the candidate word list based on the combination of 
        sentence similarities and a pretrained language model. Output the top K scored new 'BeamUnit', in a list.
        
        Input:
            stopbyLMeos: whether to use the LM '<eos>' to solely decide end of sentence, i.e. when '<eos>' gets the highest probability from the LM, remove the generated sentence out of beam. Default: False.
        
        Note:
        'word_list', 'subvocab', and 'clustermask' should be coupled, sorted based on the full vocabulary.
        """
        
        ## when no candidate word list is provided, use the full vocabulary
        if word_list is None:
            word_list = self.vocab.itos
            subvocab=None
            clustermask=None
        
        ## calculate the similarity scores
        endbus = []                         # finished sequences 
        onbeam_ids = list(range(len(bulist)))        # keep track of sequences on beam that have not aligned to the end of the source sequence
        sim_cum_allbeam = None
        indices_allbeam = None
        states_allbeam = []
        for (i, bu) in enumerate(bulist):
            try:
                scores, indices, states = simScoreNext(template_vec, word_list, ee,
                                                   prevs_state=bu.elmo_state, batch_size=batch_size,
                                                   prevs_align=bu.align_loc if mono else None,
                                                   normalized=normalized, elmo_layer=elmo_layer)
                scores_logprob = F.log_softmax(scores, dim=0)
                
                sim_cum_logprob = scores_logprob + torch.tensor(bu.sim_score, dtype=torch.float, device=self.device)
                
                sim_cum_allbeam = sim_cum_logprob if sim_cum_allbeam is None else torch.cat([sim_cum_allbeam, sim_cum_logprob])
                indices_allbeam = indices if indices_allbeam is None else torch.cat([indices_allbeam, indices])
                states_allbeam = states_allbeam + states
                
            # current sequence already aligned to the end: move out of beam
            except AssertionError as e:
                print('AssertionError:', e)
                endbus.append((i, bu.seq_len, bu.score, bu.sim_score, bu.lm_score))
                onbeam_ids.remove(i)
        
        ## calculate the RNN LM scores 
        ## note that LMModel should have the same vocab as that in Beam()
        if len(bulist) == 1 and bulist[0].word_id is None:
            # first beam step after initialization, only relying on similarity scores and no LM calculation is needed
            scores_comb = sim_cum_allbeam
            lm_cum_logprob = torch.zeros_like(sim_cum_allbeam)
            hn = [None] * len(onbeam_ids)        # at the initial step, 'onbeam_ids' wouldn't be empty anyway
        else:
            ## all sequences have aligned to the end of source sentence
            if onbeam_ids == []:
                return [], endbus
            ## do the RNN LM forward calculation
            if bulist[onbeam_ids[0]].lm_state is None:
                # 'lm_state' for the current beam is either all 'None' or all not 'None'.
                batch_hn = None
            else:
                batch_hn = (torch.cat([bulist[i].lm_state[0] for i in onbeam_ids], dim=1),
                           torch.cat([bulist[i].lm_state[1] for i in onbeam_ids], dim=1))    
            batch_text = next(LMModel.parameters()).new_tensor([bulist[i].word_id for i in onbeam_ids], dtype=torch.long).unsqueeze(0)
            subprobs, probs, hn = prob_next_1step(LMModel, batch_text, hn=batch_hn,
                                                  subvocab=subvocab, clustermask=clustermask, onscore=False, renorm=renorm,
                                                  temperature=temperature)
            
            ### LM predictes '<eos>' with the highest probability: move out of beam
            if stopbyLMeos:
                subprobs_max, subprobs_maxids = torch.max(subprobs, dim=1)
                eospos = (subprobs_maxids == word_list.index('<eos>')).nonzero()
                if eospos.size(0) > 0:        # number of ended sentences
                    # Note: have to delete backwards! Otherwise the indices will change.
                    oob_ids = [onbeam_ids.pop(ep.item()) for ep in eospos.squeeze(1).sort(descending=True)[0]]
                    oob_ids = sorted(oob_ids)
                    print('-' * 5 + ' <eos> predicted most likely by LM at location:', *oob_ids)
                    for i in oob_ids:
                        endbus.append((i, bulist[i].seq_len, bulist[i].score, bulist[i].sim_score, bulist[i].lm_score))
                    # all sequences have been predicted with '<eos>' having highest probabilities
                    if onbeam_ids == []:
                        return [], endbus
                    else:
                        remainpos = [i for i in range(len(subprobs)) if i not in eospos]
                        subprobs = subprobs[remainpos, :]
                        probs = probs[remainpos, :]
                        hn = (hn[0][:, remainpos, :], hn[1][:, remainpos, :])
                        remainpos_simallbeam = []
                        for rp in remainpos:
                            remainpos_simallbeam += list(range(len(word_list) * rp, len(word_list) * (rp + 1)))
                        sim_cum_allbeam = sim_cum_allbeam[remainpos_simallbeam]
                        indices_allbeam = indices_allbeam[remainpos_simallbeam]
                        states_allbeam = [s for (i, s) in enumerate(states_allbeam) if i in remainpos_simallbeam]
                        
            # convert the hidden state tuple into a list of tuples, corresponding to each beam sequence
            hn = list(zip(torch.chunk(hn[0], chunks=len(onbeam_ids), dim=1), torch.chunk(hn[1], chunks=len(onbeam_ids), dim=1)))
            lm_cum_logprob = subprobs.new_tensor([bulist[i].lm_score for i in onbeam_ids]).unsqueeze(1) + torch.log(subprobs)
            lm_cum_logprob = lm_cum_logprob.view(-1)      # this is the cumulative log probabilities
            
            if ifadditive:
                scores_comb = torch.log((1 - alpha) * torch.exp(sim_cum_allbeam) + alpha * torch.exp(lm_cum_logprob))
            else:
                scores_comb = (1 - alpha) * sim_cum_allbeam + alpha * lm_cum_logprob
        
        ## rank and update
        if K > len(scores_comb):
            scores_comb_sorted, ids_sorted = scores_comb.sort(descending=True)
            nexttopKK = [BeamUnit(self.vocab.stoi[word_list[i % len(word_list)]],
                                 bulist[onbeam_ids[i // len(word_list)]].cur_loc,
                                 m,
                                 scores_comb_sorted[m].item(),
                                 bulist[onbeam_ids[i // len(word_list)]].seq_len + 1,
                                 self.vocab,
                                 sim_score=sim_cum_allbeam[i].item(),
                                 lm_score=lm_cum_logprob[i].item(),
                                 lm_state=hn[i // len(word_list)],
                                 elmo_state=states_allbeam[i],
                                 align_loc=indices_allbeam[i])
                         for (m, i) in enumerate(ids_sorted)]
        else:
            scores_comb_topK, ids_topK = scores_comb.topk(K)
            nexttopKK = [BeamUnit(self.vocab.stoi[word_list[i % len(word_list)]],
                                 bulist[onbeam_ids[i // len(word_list)]].cur_loc,
                                 m,
                                 scores_comb_topK[m].item(),
                                 bulist[onbeam_ids[i // len(word_list)]].seq_len + 1,
                                 self.vocab,
                                 sim_score=sim_cum_allbeam[i].item(),
                                 lm_score=lm_cum_logprob[i].item(),
                                 lm_state=hn[i // len(word_list)],
                                 elmo_state=states_allbeam[i],
                                 align_loc=indices_allbeam[i])
                        for (m, i) in enumerate(ids_topK)]
        
        return nexttopKK, endbus
    
    def combscoreK_GPT2(self, bulist, K, template_vec, GPT2_tokenizer, GPT2_model, LMModel,
                  word_list=None, subvocab=None, clustermask=None,
                  mono=True, normalized=True, renorm=False, temperature=1,
                  bpe2word='last', alpha=0.01, stopbyLMeos=False, ifadditive=False):
        """
        Given a list of 'BeamUnit', score the next tokens from the candidate word list based on the combination of 
        sentence similarities and a pretrained language model. Output the top K scored new 'BeamUnit', in a list.
        
        Input:
            stopbyLMeos: whether to use the LM '<eos>' to solely decide end of sentence, i.e. when '<eos>' gets the highest probability from the LM, remove the generated sentence out of beam. Default: False.
        
        Note:
        'word_list', 'subvocab', and 'clustermask' should be coupled, sorted based on the full vocabulary.
        """
        
        ## when no candidate word list is provided, use the full vocabulary
        if word_list is None:
            word_list = self.vocab.itos
            subvocab=None
            clustermask=None
        
        ## calculate the similarity scores
        endbus = []                         # finished sequences 
        onbeam_ids = list(range(len(bulist)))        # keep track of sequences on beam that have not aligned to the end of the source sequence
        sim_cum_allbeam = None
        indices_allbeam = None
        states_allbeam = []
        for (i, bu) in enumerate(bulist):
            try:
                scores, indices, states = simScoreNext_GPT2(template_vec, word_list, GPT2_tokenizer, GPT2_model,
                                                   prevs_state=bu.gpt2_state,
                                                   prevs_align=bu.align_loc if mono else None,
                                                   normalized=normalized, bpe2word=bpe2word)
                scores_logprob = F.log_softmax(scores, dim=0)
                
                sim_cum_logprob = scores_logprob + torch.tensor(bu.sim_score, dtype=torch.float, device=self.device)
                
                sim_cum_allbeam = sim_cum_logprob if sim_cum_allbeam is None else torch.cat([sim_cum_allbeam, sim_cum_logprob])
                indices_allbeam = indices if indices_allbeam is None else torch.cat([indices_allbeam, indices])
                states_allbeam = states_allbeam + states
                
            # current sequence already aligned to the end: move out of beam
            except AssertionError as e:
                print('AssertionError:', e)
                endbus.append((i, bu.seq_len, bu.score, bu.sim_score, bu.lm_score))
                onbeam_ids.remove(i)
        
        ## calculate the RNN LM scores 
        ## note that LMModel should have the same vocab as that in Beam()
        if len(bulist) == 1 and bulist[0].word_id is None:
            # first beam step after initialization, only relying on similarity scores and no LM calculation is needed
            scores_comb = sim_cum_allbeam
            lm_cum_logprob = torch.zeros_like(sim_cum_allbeam)
            hn = [None] * len(onbeam_ids)        # at the initial step, 'onbeam_ids' wouldn't be empty anyway
        else:
            ## all sequences have aligned to the end of source sentence
            if onbeam_ids == []:
                return [], endbus
            ## do the RNN LM forward calculation
            if bulist[onbeam_ids[0]].lm_state is None:
                # 'lm_state' for the current beam is either all 'None' or all not 'None'.
                batch_hn = None
            else:
                batch_hn = (torch.cat([bulist[i].lm_state[0] for i in onbeam_ids], dim=1),
                           torch.cat([bulist[i].lm_state[1] for i in onbeam_ids], dim=1))    
            batch_text = next(LMModel.parameters()).new_tensor([bulist[i].word_id for i in onbeam_ids], dtype=torch.long).unsqueeze(0)
            subprobs, probs, hn = prob_next_1step(LMModel, batch_text, hn=batch_hn,
                                                  subvocab=subvocab, clustermask=clustermask, onscore=False, renorm=renorm,
                                                  temperature=temperature)
            
            ### LM predictes '<eos>' with the highest probability: move out of beam
            if stopbyLMeos:
                subprobs_max, subprobs_maxids = torch.max(subprobs, dim=1)
                eospos = (subprobs_maxids == word_list.index('<eos>')).nonzero()
                if eospos.size(0) > 0:        # number of ended sentences
                    # Note: have to delete backwards! Otherwise the indices will change.
                    oob_ids = [onbeam_ids.pop(ep.item()) for ep in eospos.squeeze(1).sort(descending=True)[0]]
                    oob_ids = sorted(oob_ids)
                    print('-' * 5 + ' <eos> predicted most likely by LM at location:', *oob_ids)
                    for i in oob_ids:
                        endbus.append((i, bulist[i].seq_len, bulist[i].score, bulist[i].sim_score, bulist[i].lm_score))
                    # all sequences have been predicted with '<eos>' having highest probabilities
                    if onbeam_ids == []:
                        return [], endbus
                    else:
                        remainpos = [i for i in range(len(subprobs)) if i not in eospos]
                        subprobs = subprobs[remainpos, :]
                        probs = probs[remainpos, :]
                        hn = (hn[0][:, remainpos, :], hn[1][:, remainpos, :])
                        remainpos_simallbeam = []
                        for rp in remainpos:
                            remainpos_simallbeam += list(range(len(word_list) * rp, len(word_list) * (rp + 1)))
                        sim_cum_allbeam = sim_cum_allbeam[remainpos_simallbeam]
                        indices_allbeam = indices_allbeam[remainpos_simallbeam]
                        states_allbeam = [s for (i, s) in enumerate(states_allbeam) if i in remainpos_simallbeam]
                        
            # convert the hidden state tuple into a list of tuples, corresponding to each beam sequence
            hn = list(zip(torch.chunk(hn[0], chunks=len(onbeam_ids), dim=1), torch.chunk(hn[1], chunks=len(onbeam_ids), dim=1)))
            lm_cum_logprob = subprobs.new_tensor([bulist[i].lm_score for i in onbeam_ids]).unsqueeze(1) + torch.log(subprobs)
            lm_cum_logprob = lm_cum_logprob.view(-1)      # this is the cumulative log probabilities
            
            if ifadditive:
                scores_comb = torch.log((1 - alpha) * torch.exp(sim_cum_allbeam) + alpha * torch.exp(lm_cum_logprob))
            else:
                scores_comb = (1 - alpha) * sim_cum_allbeam + alpha * lm_cum_logprob
        
        ## rank and update
        if K > len(scores_comb):
            scores_comb_sorted, ids_sorted = scores_comb.sort(descending=True)
            nexttopKK = [BeamUnit(self.vocab.stoi[word_list[i % len(word_list)]],
                                 bulist[onbeam_ids[i // len(word_list)]].cur_loc,
                                 m,
                                 scores_comb_sorted[m].item(),
                                 bulist[onbeam_ids[i // len(word_list)]].seq_len + 1,
                                 self.vocab,
                                 sim_score=sim_cum_allbeam[i].item(),
                                 lm_score=lm_cum_logprob[i].item(),
                                 lm_state=hn[i // len(word_list)],
                                 gpt2_state=states_allbeam[i],
                                 align_loc=indices_allbeam[i])
                         for (m, i) in enumerate(ids_sorted)]
        else:
            scores_comb_topK, ids_topK = scores_comb.topk(K)
            nexttopKK = [BeamUnit(self.vocab.stoi[word_list[i % len(word_list)]],
                                 bulist[onbeam_ids[i // len(word_list)]].cur_loc,
                                 m,
                                 scores_comb_topK[m].item(),
                                 bulist[onbeam_ids[i // len(word_list)]].seq_len + 1,
                                 self.vocab,
                                 sim_score=sim_cum_allbeam[i].item(),
                                 lm_score=lm_cum_logprob[i].item(),
                                 lm_state=hn[i // len(word_list)],
                                 gpt2_state=states_allbeam[i],
                                 align_loc=indices_allbeam[i])
                        for (m, i) in enumerate(ids_topK)]
        
        return nexttopKK, endbus
                
   
