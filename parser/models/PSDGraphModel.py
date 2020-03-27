#!/usr/bin/env python3.6
# coding=utf-8
'''

Deep Learning Models for variational inference of alignment.
Posterior , LikeliHood helps computing posterior weighted likelihood regarding relaxation.

Also the whole PSD model is combined here.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import numpy as np
from parser.models.ConceptModel import *
from parser.models.MultiPassRelModel import *
from parser.modules.encoder_zoo import *

from parser.modules.GumbelSoftMax import renormalize,sink_horn,gumbel_noise_sample
from parser.modules.helper_module import *

from allennlp.modules.seq2seq_encoders.stacked_self_attention import *
import json

from copy import deepcopy
import logging
logger = logging.getLogger("mrp")

#Encoding linearized PSD concepts for vartiaonal alignment model
class PSDTransformerEncoder(nn.Module):

    def __init__(self, opt, embs, transformer_encoder):
        super(PSDTransformerEncoder, self).__init__()
        # transformer configuration:
        # opt.psd_transformer_config = "hidden_dim:projection_dim:feedforward_hidden_dim:num_layers:num_atttention_heads:use_positional_encoding:dropout_prob:residual_dropout_prob:attention_dropout_prob"
        self.transformer_encoder = transformer_encoder
        # hidden_dim for stacked_self_att
        inputSize = embs["psd_target_pos_lut"].embedding_dim +  embs["lemma_lut"].embedding_dim
        assert self.transformer_encoder.get_input_dim() == inputSize, "inputSize not match {} != actual encoder_size{}".format(self.transformer_encoder.get_input_dim(), inputSize)
        self.amr_enc_size = self.transformer_encoder.get_output_dim()
        self.psd_target_pos_lut = embs["psd_target_pos_lut"]
        self.lemma_lut  = embs["lemma_lut"]
        self.alpha_dropout = opt.alpha_dropout
        if opt.cuda:
            self.device_type = 'cuda'
            self.cuda()

    #input:len,  batch, n_feature
    #output: len, batch, hidden_size * num_directions
    def forward(self, packed_input, hidden=None):
        assert isinstance(packed_input,PackedSequence)
        input = packed_input.data

        if self.alpha_dropout and self.training:
            input = data_dropout(input, self.alpha_dropout)

        pos_embed = self.psd_target_pos_lut(input[:,PSD_POS])
        lemma_embed = self.lemma_lut(input[:,PSD_LE])

        emb = torch.cat([pos_embed,lemma_embed],1) #  len,batch,embed
        packed_emb = PackedSequence(emb, packed_input.batch_sizes)
        # padded_emb is padded tensor, batch x len x n_feture, lengths
        padded_emb, lengths = unpack(packed_emb, batch_first = True)
        max_len = padded_emb.size(1)
        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        if self.device_type == 'cuda':
            mask = mask.cuda()
        # logger.info("mask information is {}".format(mask))
        outputs = self.transformer_encoder(padded_emb, mask)
        # shape (batch_size, sequence_length, hidden_dim)
        return  pack(outputs, lengths, batch_first = True)

#Encoding linearized PSD concepts for vartiaonal alignment model
class PSDEncoder(nn.Module):

    def __init__(self, opt, embs):
        self.layers = opt.amr_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.amr_rnn_size % self.num_directions == 0
        self.hidden_size = opt.amr_rnn_size // self.num_directions
        inputSize = embs["psd_target_pos_lut"].embedding_dim +  embs["lemma_lut"].embedding_dim
        self.amr_enc_size = opt.amr_rnn_size
        super(PSDEncoder, self).__init__()

        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=opt.amr_enlayers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

        self.psd_target_pos_lut = embs["psd_target_pos_lut"]
        self.lemma_lut  = embs["lemma_lut"]

        # self.alpha_dropout = nn.AlphaDropout(opt.alpha_dropout)  #unk with alpha
        self.alpha_dropout = opt.alpha_dropout
        if opt.cuda:
            self.cuda()

    #input:len,  batch, n_feature
    #output: len, batch, hidden_size * num_directions
    def forward(self, packed_input, hidden=None):
        assert isinstance(packed_input,PackedSequence)
        input = packed_input.data

        if self.alpha_dropout and self.training:
            input = data_dropout(input, self.alpha_dropout)

        pos_embed = self.psd_target_pos_lut(input[:,PSD_POS])
        lemma_embed = self.lemma_lut(input[:,PSD_LE])

        emb = torch.cat([pos_embed,lemma_embed],1) #  len,batch,embed
        emb = PackedSequence(emb, packed_input.batch_sizes)
        outputs, _ = self.rnn(emb, hidden)
        return  outputs
