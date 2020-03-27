#!/usr/bin/env python3.6
# coding=utf-8
'''

Deep Learning Models for concept identification

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import torch
import torch.nn as nn
from parser.modules.helper_module import *
from parser.modules.encoder_zoo import *
from torch.nn.utils.rnn import PackedSequence
from utility.constants import *
from pytorch_transformers.modeling_bert import BertModel
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn import util
import logging
logger = logging.getLogger("mrp.PSDConceptModel")

class PSD_Concept_Classifier(nn.Module):

    def __init__(self, opt, embs, concept_src_enc_size):
        super(PSD_Concept_Classifier, self).__init__()
        self.concept_src_enc_size = concept_src_enc_size

        # the size of embeddings, contains a UNK category
        self.n_target_pos = embs["psd_target_pos_lut"].num_embeddings
        # the size of embeddings, contains a UNK aux term
        self.n_sense = embs["psd_sense_lut"].num_embeddings
        # the size of embeddings, contains a UNK high term
        self.n_high = embs["psd_high_lut"].num_embeddings

        self.target_pos_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.concept_src_enc_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_target_pos,bias = opt.psd_target_pos_bias))

        self.sense_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.concept_src_enc_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_sense,bias = opt.psd_sense_bias))

        self.le_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.concept_src_enc_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_high+1,bias = opt.lemma_bias))

        self.t = 1
        self.sm = nn.Softmax(dim=-1)
        if opt.cuda:
            self.cuda()


    def forward(self, src_enc ):
        '''
            src_enc: pack(data x txt_rnn_size ,batch_size)
           src_le:  pack(data x 1 ,batch_size)

           out:  (datax n_target_pos, batch_size),    (data x n_high+1,batch_size)
        '''

        assert isinstance(src_enc,PackedSequence)


     #   high_embs = self.amr_high_lut.weight.expand(le_score.size(0),self.n_high,self.dim)
      #  le_self_embs = self.lemma_lut(src_le.data).unsqueeze(1)
      #  le_emb = torch.cat([high_embs,le_self_embs],dim=1) #data x sense x dim

        pre_enc =src_enc.data

        target_pos_score = self.target_pos_score(pre_enc) #  n_data x n_cat
        target_pos_prob = self.sm(target_pos_score)
        sense_score = self.sense_score(pre_enc)
        sense_prob = self.sm(sense_score)
        le_score = self.le_score(pre_enc)
        le_prob = self.sm(le_score)
        batch_sizes = src_enc.batch_sizes
        # PackedSequence https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        # follow the order, PSD_POS = 0, PSD_LE= 1, PSD_SENSE = 2
        return PackedSequence(target_pos_prob,batch_sizes), PackedSequence(le_prob,batch_sizes), PackedSequence(sense_prob,batch_sizes)
