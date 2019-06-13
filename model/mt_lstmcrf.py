#
#@author: Shi Ke
#

import torch
import torch.nn as nn
import numpy as np
from model.lstmcrf import NNCRF


class MT_LSTMCRF(nn.Module):

    def __init__(self, config_base, config_conll, config_ontonotes):
        super(MT_LSTMCRF, self).__init__()
        self.config_base = config_base
        self.config_conll = config_conll
        self.config_ontonotes = config_ontonotes
        self.lstmcrf_base = NNCRF(config_base)
        self.lstmcrf_conll = NNCRF(config_conll)
        self.lstmcrf_ontonotes = NNCRF(config_ontonotes)

    def neg_log_obj_total(self, words, word_seq_lens, batch_context_emb, chars, char_seq_lens, prefix_label, conll_label, notes_label, mask_base, mask_conll, mask_ontonotes):
        loss_base, hiddens_base = self.lstmcrf_base.neg_log_obj(words, word_seq_lens, batch_context_emb, chars, char_seq_lens, prefix_label, mask_base)
        # hidden_base = w1 * h1
        loss_conll, _ = self.lstmcrf_conll.neg_log_obj(words, word_seq_lens, batch_context_emb, chars, char_seq_lens, conll_label, mask_conll, hiddens_base)
        loss_ontonotes, _ = self.lstmcrf_ontonotes.neg_log_obj(words, word_seq_lens, batch_context_emb, chars, char_seq_lens, notes_label, mask_ontonotes, hiddens_base)
        loss_total = loss_base + loss_conll + loss_ontonotes
        # loss_total = loss_ontonotes
        return loss_total

    def decode(self, batchinput):
        words, word_seq_lens, batch_context_emb, chars, char_seq_lens, prefix_label, conll_label, notes_label, mask_base, mask_conll, mask_ontonotes = batchinput
        _, hiddens_base = self.lstmcrf_base.neg_log_obj(words, word_seq_lens, batch_context_emb, chars, char_seq_lens, prefix_label, mask_base)
        bestScores_conll, decodeIdx_conll = self.lstmcrf_conll.decode(batchinput, hiddens_base)
        bestScores_notes, decodeIdx_notes = self.lstmcrf_ontonotes.decode(batchinput, hiddens_base)

        return bestScores_conll, decodeIdx_conll, bestScores_notes, decodeIdx_notes, mask_conll, mask_ontonotes