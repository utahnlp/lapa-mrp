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
logger = logging.getLogger("mrp.DMConceptModel")


class DM_Concept_Classifier(nn.Module):

    def __init__(self, opt, embs, concept_src_enc_size):
        super(DM_Concept_Classifier, self).__init__()
        self.concept_src_enc_size = concept_src_enc_size

        # the size of embeddings, contains a UNK category
        self.n_target_pos = embs["dm_target_pos_lut"].num_embeddings
        # the size of embeddings, contains a UNK high term
        self.n_cat_high = embs["dm_high_lut"].num_embeddings
        ## the size of embeddings, contains a UNK high term
        #self.n_le_high = embs["dm_high_le_lut"].num_embeddings
        # the size of embeddings, contains a UNK aux term
        self.n_sense = embs["dm_sense_lut"].num_embeddings

        self.target_pos_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.concept_src_enc_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_target_pos,bias = opt.dm_target_pos_bias))

        self.cat_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.concept_src_enc_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_cat_high+1, bias = opt.dm_cat_bias))

              # not we don't use predicate sense for now
        self.sense_score =nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.concept_src_enc_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_sense,bias = opt.dm_sense_bias))

        #self.le_score =nn.Sequential(
        #    nn.Dropout(opt.dropout),
        #    nn.Linear(self.concept_src_enc_size, 256),
        #    nn.ReLU(),
        #    nn.Linear(256, self.n_le_high +1,bias = opt.dm_sense_bias))

        self.t = 1
        self.sm = nn.Softmax(dim=-1)
        if opt.cuda:
            self.cuda()


    def forward(self, src_enc ):
        '''
            src_enc: pack(data x txt_rnn_size ,batch_size)
           src_le:  pack(data x 1 ,batch_size)

           out:  (datax n_cat_high, batch_size),    (data x n_high+1,batch_size)
        '''

        assert isinstance(src_enc,PackedSequence)


     #   high_embs = self.amr_high_lut.weight.expand(le_score.size(0),self.n_high,self.dim)
      #  le_self_embs = self.lemma_lut(src_le.data).unsqueeze(1)
      #  le_emb = torch.cat([high_embs,le_self_embs],dim=1) #data x high+1 x dim

        pre_enc =src_enc.data

        target_pos_score = self.target_pos_score(pre_enc) #  n_data x n_cat_high
        target_pos_prob = self.sm(target_pos_score)
        cat_score = self.cat_score(pre_enc)#  n_data x n_cat_high
        cat_prob = self.sm(cat_score)
        sense_score = self.sense_score(pre_enc)
        sense_prob = self.sm(sense_score)
        #le_score = self.le_score(pre_enc)#  n_data x n_cat_high
        #le_prob = self.sm(le_score)
        batch_sizes = src_enc.batch_sizes
        # PackedSequence https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        #return PackedSequence(target_pos_prob,batch_sizes),PackedSequence(cat_prob,batch_sizes),PackedSequence(sense_prob,batch_sizes),PackedSequence(le_prob,batch_sizes)
        return PackedSequence(target_pos_prob,batch_sizes),PackedSequence(cat_prob,batch_sizes),PackedSequence(sense_prob,batch_sizes)
