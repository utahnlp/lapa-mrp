#!/usr/bin/env python3.6
# coding=utf-8
'''

Deep Learning Models for relation identification

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import torch
import torch.nn as nn
from parser.modules.helper_module import mypack ,myunpack,MyPackedSequence,MyDoublePackedSequence,mydoubleunpack,mydoublepack,DoublePackedSequence,doubleunpack
from parser.models.ConceptModel import *
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from utility.constants import *
import logging
logger = logging.getLogger("amr.EDSRelModel")

#combine eds node embedding and aligned sentence token embedding
class EDSRootEncoder(nn.Module):

    def __init__(self, opt, embs, root_src_enc_size):
        super(EDSRootEncoder, self).__init__()
        #TODO: for eds layers
        self.layers = opt.amr_enlayers
        #share hyper parameter with relation model
        self.size = opt.rel_dim
        self.root_src_enc_size = root_src_enc_size

        # inputSize = embs["eds_target_pos_lut"].embedding_dim + embs["eds_cat_lut"].embedding_dim+root_src_enc_size
        inputSize = embs["eds_cat_lut"].embedding_dim+root_src_enc_size

        self.eds_cat_lut  = embs["eds_cat_lut"]

        self.root = nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(inputSize,self.size ),
            nn.ReLU()
        )

        # self.alpha_dropout = nn.AlphaDropout(p = opt.alpha_dropout)
        self.alpha_dropout = opt.alpha_dropout
        if opt.cuda:
            self.cuda()

    def getEmb(self,indexes,src_enc):
        """
        # index:  rel_index_batch:  list(batch, real_gold_amr_len), but content is the index of recatogrized amr index, is a mapping
        # src_enc: is weighted root_src_enc, MyPackedSequence(data: packed_re_amr_len x txt_enc_size, lengtgs: re_amr_lens)
        """
        head_emb,lengths = [],[]
        src_enc = myunpack(*src_enc)  #  list(batch, real_re_amr_len x src_enc_size)
        for i, index in enumerate(indexes): # indexse, batch_size, real_gold_amr_lens
            enc = src_enc[i]  #real_re_amr_len x src_enc_size,
            head_emb.append(enc[index])  #the content of index if real_re_amd_index, list(batch, real_gold_amr_len, dim)
            lengths.append(len(index))   #list(batch, real_gold_amr_len)
        return mypack(head_emb,lengths)  # MyPackedSequence(data: packed_gold_amr_len x dim, lengths: readl_gold_amr_len)

    def forward(self, input, index,src_enc):
        """
        # input: relBatch, packed_gold_amr_lengths x n_feature,  AMR_CAT, AMR_LE, AMR_NER, AMR_SENSE, index of nodes,
        # index:  rel_index_batch:  list(batch, real_gold_amr_len), but content is the index of recatogrized amr index, is a mapping
        # src_enc: is weighted root_src_enc, MyPackedSequence(data: packed_re_amr_len x txt_enc_size, lengtgs: re_amr_lens)
        """
        assert isinstance(input, MyPackedSequence),input
        # lengths: real_gold_amr_length_
        input,lengths = input
        if self.alpha_dropout and self.training:
            input = data_dropout(input, self.alpha_dropout)
        eds_cat_embed = self.eds_cat_lut(input[:,EDS_CAT])

        # eds_emb = torch.cat([eds_target_pos_embed,eds_cat_embed],1)
        eds_emb = torch.cat([eds_cat_embed],1)

        # head_emb: MyPackedSequence(data: packed_gold_amr_len x dim, lengths: readl_gold_amr_len)
        head_emb = self.getEmb(index,src_enc)  #packed, mydoublepacked

        # head_emb.data, [packed_head_amr_len, src_enc_size]
        root_emb = torch.cat([eds_emb,head_emb.data],1)
        root_emb = self.root(root_emb)

        return MyPackedSequence(root_emb,lengths)

#combine eds node embedding and aligned sentence token embedding
class EDSRelEncoder(nn.Module):

    def __init__(self, opt, embs, rel_src_enc_size):
        super(EDSRelEncoder, self).__init__()

        # TODO: for eds layers
        self.layers = opt.amr_enlayers
        self.size = opt.rel_dim
        self.rel_src_enc_size = rel_src_enc_size

        # self.inputSize = embs["eds_target_pos_lut"].embedding_dim + embs["eds_cat_lut"].embedding_dim+ self.rel_src_enc_size
        self.inputSize = embs["eds_cat_lut"].embedding_dim +  self.rel_src_enc_size

        self.head = nn.Sequential(
            nn.Dropout(opt.dropout),
          nn.Linear(self.inputSize,self.size )
        )

        self.dep = nn.Sequential(
            nn.Dropout(opt.dropout),
          nn.Linear(self.inputSize,self.size )
        )

        self.eds_cat_lut = embs["eds_cat_lut"]
        # self.alpha_dropout = nn.AlphaDropout(opt.alpha_dropout)
        self.alpha_dropout = opt.alpha_dropout

        if opt.cuda:
            self.cuda()

    def getEmb(self,indexes,src_enc):
        """
        # index:  rel_index_batch:  list(batch, real_gold_amr_len), but content is the index of recatogrized amr index, is a mapping
        # src_enc: DoublePackedSequence(packed: packed_g_amr_len x max_re_amr_len x dim), length= re_lens), out_l = g_amr_len
        """
        head_emb,dep_emb = [],[]
        # unpacked_src_enc: list(batch, re_g_real_len x max_re_amr_len x dim )
        unpacked_src_enc, _ = doubleunpack(src_enc)
        length_pairs = []
        for i, index in enumerate(indexes): # indexes : list(batch, real_g_amr_len)
            enc = unpacked_src_enc[i]  #g_amr_real_len x  max_re_amr_l x dim
            dep_emb.append(enc.index_select(1,index))  #list(batch_size, real_g_amr_l x real_g_amr_l x dim)
            # head_index : [real_g_amr_len, 1, 1] -> [real_g_amr_len, 1, dim]
            head_index = index.unsqueeze(1).unsqueeze(2).expand(enc.size(0),1,enc.size(-1))
            # logger.info("enc: {}, head_idnex :{}".format(enc, head_index))
            head_emb.append(enc.gather(1,head_index).squeeze(1))  # list(batch, real_g_amr_l x dim)
            length_pairs.append([len(index),len(index)])
        # head_emb_t :  MyPackedSequence(data: packed_real_g_amr_l x dim), g_amr_l
        # dep_emb_t :  MyDoublePackedSequence(PackedSequenceLength(packed_real_g_amr_l x real_g_amr_l x dim), length_pairs)
        # length_pairs :(g_amr_l, g_amr_l)
        return mypack(head_emb,[ls[0] for ls in length_pairs]),mydoublepack(dep_emb,length_pairs),length_pairs

    def forward(self, input, index,src_enc):
        """
        # input: relBatch: packed_gold_amr_lengths x n_feature,  AMR_CAT, AMR_LE, AMR_NER, AMR_SENSE, index of nodes,
        #  mypacked_seq[packed_batch_gold_amr_len x tgt_feature_dim]
        # index:  rel_index_batch:  list(batch, real_gold_amr_len), but content is the index of recatogrized amr index, is a mapping
        # src_enc: DoublePackedSequence(packed: packed_g_amr_len x re_amr_len x dim), length= re_lens)
        """
        assert isinstance(input, MyPackedSequence),input
        # lengths: real_gold_amr_lens
        # after unpack, input is packed_gold_amr_lengths x n_features
        input,lengths = input
        if self.alpha_dropout and self.training:
            input = data_dropout(input, self.alpha_dropout)

        eds_cat_embed = self.eds_cat_lut(input[:,EDS_CAT])
        eds_emb = torch.cat([eds_cat_embed],1)

        # head_emb_t :  MyPackedSequence(data: packed_real_g_amr_l x dim), g_amr_l
        # dep_emb_t :  MyDoublePackedSequence(PackedSequenceLength(packed_real_g_amr_l x real_g_amr_l x dim), length_pairs)
        # length_pairs :(g_amr_l, g_amr_l)
        head_emb_t,dep_emb_t,length_pairs = self.getEmb(index,src_enc)  #packed, mydoublepacked

        head_emb = torch.cat([eds_emb,head_emb_t.data],1)

        dep_eds_emb_t = myunpack(*MyPackedSequence(eds_emb,lengths))
        dep_eds_emb = [ emb.unsqueeze(0).expand(emb.size(0),emb.size(0),emb.size(-1))      for emb in dep_eds_emb_t]

        mydouble_eds_emb = mydoublepack(dep_eds_emb,length_pairs)

        dep_emb = torch.cat([mydouble_eds_emb.data,dep_emb_t.data],-1)

        # emb_unpacked = myunpack(emb,lengths)
        assert head_emb.size(-1) == self.inputSize, "wrong head  size {}".format(head_emb.size())
        # head_packed :  MyPackedSequence(data: packed_real_g_amr_l x rel_dim), g_amr_l
        head_packed = MyPackedSequence(self.head(head_emb),lengths) #  total,rel_dim
        head_eds_packed = MyPackedSequence(eds_emb,lengths) #  total,rel_dim

        size = dep_emb.size()
        assert dep_emb.size(-1) == self.inputSize, "wrong dep size {}".format(dep_emb.size())
        dep = self.dep(dep_emb.view(-1,size[-1])).view(size[0],size[1],-1)

        # dep_emb_t :  MyDoublePackedSequence(PackedSequenceLength(packed_real_g_amr_l x real_g_amr_l x rel_dim), length_pairs)
        dep_packed  = MyDoublePackedSequence(MyPackedSequence(dep,mydouble_eds_emb[0][1]),mydouble_eds_emb[1],dep)

        return  head_eds_packed,head_packed,dep_packed  #,MyPackedSequence(emb,lengths)
