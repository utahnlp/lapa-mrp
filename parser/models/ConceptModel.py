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
from parser.models.AMRConceptModel import *
from parser.models.DMConceptModel import *
from parser.models.PSDConceptModel import *
from parser.modules.encoder_zoo import *
from torch.nn.utils.rnn import PackedSequence
from utility.constants import *
from pytorch_transformers.modeling_bert import BertModel
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn import util
import logging
logger = logging.getLogger("mrp.ConceptModel")

class ConceptIdentifier(nn.Module):
    #could share encoder with other model
    def __init__(self, opt, embs, encoder = None, component_dict={}, frame="amr"):
        super(ConceptIdentifier, self).__init__()
        self.component_dict = component_dict
        # it is for shared bert_model
        if encoder:
            self.snt_encoder = encoder
        else:
            self.snt_encoder = EncoderZoo.create_sentence_encoder_from_config(embs, self.component_dict, opt.concept_snt_encoder, opt)
        if frame == "amr":
            self.generator = AMR_Concept_Classifier( opt, embs, concept_src_enc_size = self.snt_encoder.src_enc_size)
        elif frame == "dm":
            self.generator = DM_Concept_Classifier( opt, embs, concept_src_enc_size = self.snt_encoder.src_enc_size)
        elif frame == "psd":
            self.generator = PSD_Concept_Classifier( opt, embs, concept_src_enc_size = self.snt_encoder.src_enc_size)
        else:
            raise NotImplementedError("{} is not supported now".format(frame))

    def forward(self, srcBatch, src_charBatch, bertBatch = None, bertIndexBatch = None):
        # src_enc: encoding for each token, shape (seq_len x batch, num_directions * hidden_size)
        src_enc = self.snt_encoder(srcBatch, src_charBatch, bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)
        # probBatch : pack(data * cat), pack(data*le), pack(data*ner)
        probBatch = self.generator(src_enc)
        return probBatch,src_enc
