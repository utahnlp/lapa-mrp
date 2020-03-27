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
import logging
logger = logging.getLogger("mrp.AMRConceptModel")

class AMR_Concept_Classifier(nn.Module):

    def __init__(self, opt, embs, concept_src_enc_size):
        super(AMR_Concept_Classifier, self).__init__()
        self.concept_src_enc_size = concept_src_enc_size

        # the size of embeddings, contains a UNK category
        self.n_cat = embs["amr_cat_lut"].num_embeddings
        # the size of embeddings, contains a UNK high term
        self.n_high = embs["amr_high_lut"].num_embeddings
        # the size of embeddings, contains a UNK aux term
        self.n_aux = embs["amr_aux_lut"].num_embeddings

        self.cat_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.concept_src_enc_size, self.n_cat,bias = opt.cat_bias))

        self.le_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.concept_src_enc_size, self.n_high+1,bias = opt.lemma_bias))

        self.ner_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.concept_src_enc_size, self.n_aux,bias = opt.cat_bias))

        self.t = 1
        self.sm = nn.Softmax(dim=-1)
        if opt.cuda:
            self.cuda()



    def forward(self, src_enc ):
        '''
            src_enc: pack(data x txt_rnn_size ,batch_size)
           src_le:  pack(data x 1 ,batch_size)

           out:  (datax n_cat, batch_size),    (data x n_high+1,batch_size)
        '''

        assert isinstance(src_enc,PackedSequence)


     #   high_embs = self.amr_high_lut.weight.expand(le_score.size(0),self.n_high,self.dim)
      #  le_self_embs = self.lemma_lut(src_le.data).unsqueeze(1)
      #  le_emb = torch.cat([high_embs,le_self_embs],dim=1) #data x high+1 x dim

        pre_enc =src_enc.data

        cat_score = self.cat_score(pre_enc) #  n_data x n_cat
        ner_score = self.ner_score(pre_enc)#  n_data x n_cat
        le_score = self.le_score(pre_enc)
        le_prob = self.sm(le_score)
        cat_prob = self.sm(cat_score)
        ner_prob = self.sm(ner_score)
        batch_sizes = src_enc.batch_sizes
        # PackedSequence https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        return   PackedSequence(cat_prob,batch_sizes),PackedSequence(le_prob,batch_sizes),PackedSequence(ner_prob,batch_sizes)
