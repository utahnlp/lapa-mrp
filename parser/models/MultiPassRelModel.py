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
from parser.models.AMRRelModel import *
from parser.models.DMRelModel import *
from parser.models.PSDRelModel import *
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import BertModel
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn import util
from parser.modules.bert_utils import *
from parser.modules.char_utils import *
from utility.constants import *
import logging
logger = logging.getLogger("amr.MultiPassRelModel")

class RelBertSentenceEncoder(nn.Module):

    def __init__(self, opt, bert_model, bert_layers_mix_indices):
        # for a single sentence encoder, type id are full of zeors
        # input mask need to be extened with lengths.
       # https://github.com/allenai/allennlp/blob/e9287d4d48d2d980226a4acf70d03eedda67c548/allennlp/modules/scalar_mix.py
        super(RelBertSentenceEncoder, self).__init__()
        self.opt = opt
        self.bert_model = bert_model
        assert self.bert_model != None, "bert_model is None"
        self.rel_src_enc_size = self.bert_model.config.hidden_size + 1
        self.mix_size  = len(bert_layers_mix_indices)
        self.bert_layers_mix_indices = bert_layers_mix_indices
        assert self.mix_size <= self.bert_model.config.num_hidden_layers and self.mix_size > 0, "wrong bert_layers_mix_indices = {}".format(bert_layer_mix_indices)
        if self.mix_size > 1:
            self._scalar_mix = ScalarMix(self.mix_size, do_layer_norm=False)
        else:
            self._scalar_mix = None

    def forward(self, packed_input: PackedSequence,src_charBatch, packed_posterior, bertBatch = None, bertIndexBatch=None):
        """
        packed_input: [packed_batch_src_len x dim], dim depends on whether it is src output encoding or input embeddin
        packed_posterior : mypack(packed_gold_amr_len  x src_len), gold_amr_l
        """
        if packed_input.data.size(-1) == self.rel_src_enc_size:
            #logger.info("Just use packed input as output:{}".format(packed_input))
            # it is already encoded
            packed_output = packed_input
        else:
            packed_output, pooled_output = BertEncoderUtils.encodeSequence(self.bert_model, packed_input, bertBatch, bertIndexBatch, self._scalar_mix, self.bert_layers_mix_indices)

        #Outputs: pack(data,lengths,batch_first=True): [packed_real_gold_amr_len x src_len, dim], src_lens
        # it take all packed_real_gold_amr_len as batch_size
        # amr_len: all real_gold_amr_lens
        Outputs, amr_len = RelPosteriorUtils.posteriorIndictedEmb(packed_output, packed_posterior)
        # assert Outputs.data.size(-1) == self.rel_src_enc_size + 1 , "wrong rel_src_enc size {}".format(Outputs.data.size())
        return  DoublePackedSequence(Outputs,amr_len,Outputs.data)

class RelPosteriorUtils:
    @staticmethod
    def posteriorIndictedEmb(embs, posterior):
        """
        embs: actually are bert encodings, padded_batch_src_len x bert_output_dim
        posterior : mypack(packed_gold_amr_len  x src_len), gold_amr_l
        """
        # after unpack, embs :[max_src_len x batch x bert_output_dim]
        embs,src_len = unpack(embs)

        if isinstance(posterior,MyPackedSequence):
            # after unpack, posterior: list(batch_size, real_gold_amr_len x src_len)
            posterior = myunpack(*posterior)
            # after tranpose, batch_size x padd_src_len x dim
            embs = embs.transpose(0,1)
            out = []
            lengths = []
            # real gold amr length for each example in the batch
            amr_len = [len(p) for p in posterior]
            for i,emb in enumerate(embs):
                # expanded_emb: real_gold_amr_len x src_len x dim
                expanded_emb = emb.unsqueeze(0).expand([amr_len[i]]+[i for i in emb.size()])
                indicator = posterior[i].unsqueeze(2)  # real_gold_amr_len x src_len x 1
                out.append(torch.cat([expanded_emb,indicator],2))  # real_gold_amr_len x src_len x (dim+1)
                # out.append(expanded_emb)  # real_gold_amr_len x src_len x (dim+1)
                lengths = lengths + [src_len[i]]*amr_len[i] # lenghs =list(batch, real_gold_amr_len]), all stores src_lens
            data = torch.cat(out,dim=0)  # packed_real_gold_amr_len x src_len x (dim+1)

            return pack(data,lengths,batch_first=True),amr_len
        elif isinstance(posterior,list):
            # real alignments
            embs = embs.transpose(0,1)
            src_l = embs.size(1)
            amr_len = [len(i) for i in posterior]
            out = []
            lengths = []
            for i,emb in enumerate(embs):
                amr_l = len(posterior[i])
                expanded_emb = emb.unsqueeze(0).expand([amr_l]+[i for i in emb.size()]) # amr_len x src_len x dim
                indicator = emb.new_zeros((amr_l,src_l))
                scattered_indicator = indicator.scatter(1, posterior[i].unsqueeze(1), 1.0) # amr_len x src_len x 1
                out.append(torch.cat([expanded_emb, scattered_indicator.unsqueeze(2)],2))  # amr_len x src_len x (dim+1)
                #out.append(expanded_emb)  # amr_len x src_len x (dim+1)
                lengths = lengths + [src_len[i]]*amr_l   # batch x amr_len, src_len
            data = torch.cat(out,dim=0)    # batch x amr_len, x src_len x (dim +1)

            return pack(data,lengths,batch_first=True),amr_len

#Encoding linearized concepts for vartiaonal alignment model
class RelTransformerSentenceEncoder(nn.Module):

    def __init__(self, opt, embs, transformer_encoder, bert_model = None, bert_layers_mix_indices= None, char_encoder = None):
        super(RelTransformerSentenceEncoder, self).__init__()
        # transformer configuration:
        assert transformer_encoder, "transformer_encoder for root is None"
        self.transformer_encoder = transformer_encoder
        inputSize =  embs["word_fix_lut"].embedding_dim + embs["lemma_lut"].embedding_dim\
                     +embs["pos_lut"].embedding_dim+embs["ner_lut"].embedding_dim+1

        self.char_encoder = char_encoder
        if self.char_encoder:
            inputSize = inputSize + self.char_encoder.get_output_dim()
            self.char_lut = embs["char_lut"]

        # adding bert
        self.bert_model = bert_model
        if self.bert_model:
            self.mix_size  = len(bert_layers_mix_indices)
            self.bert_layers_mix_indices = bert_layers_mix_indices
            assert self.mix_size <= self.bert_model.config.num_hidden_layers and self.mix_size > 0, "wrong bert_layers_mix_indices = {}".format(bert_layer_mix_indices)
            if self.mix_size > 1:
                self._scalar_mix = ScalarMix(self.mix_size, do_layer_norm=False)
            else:
                self._scalar_mix = None
            inputSize = inputSize + self.bert_model.config.hidden_size

        assert self.transformer_encoder.get_input_dim() == inputSize, "inputSize not match {} != actual encoder_size{}".format(self.transformer_encoder.get_input_dim(), inputSize)
        self.rel_src_enc_size = self.transformer_encoder.get_output_dim()
        self.lemma_lut = embs["lemma_lut"]
        self.word_fix_lut  = embs["word_fix_lut"]
        self.pos_lut = embs["pos_lut"]
        self.ner_lut = embs["ner_lut"]
        self.alpha_dropout = opt.alpha_dropout
        if opt.cuda:
            self.device_type = 'cuda'
            self.cuda()

    #input:len,  batch, n_feature
    #output: len, batch, hidden_size * num_directions
    def forward(self, packed_input, src_charBatch, packed_posterior,hidden=None,bertBatch=None, bertIndexBatch=None):
        assert isinstance(packed_input,PackedSequence)
        input = packed_input.data

        if self.alpha_dropout and self.training:
            input = data_dropout(input, self.alpha_dropout)
        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])

        emb = torch.cat([lemma_emb,pos_emb,ner_emb],1)#  data,embed
        # it only dropout the other embeb, for word embed , it did not dropout
        if self.char_encoder:
            # inputs: (src_len x batch_size, n_characters, dim)
            # mask = (src_len x batch_size, n_characters)
            packed_char_encoding = CharEncoderUtils.encodeSequence(self.char_encoder, src_charBatch, self.char_lut)
            emb = torch.cat([emb,packed_char_encoding.data],1)#  data,embed

        # add bert
        if self.bert_model:
            packed_bert_encs,_ = BertEncoderUtils.encodeSequence(self.bert_model, packed_input, bertBatch, bertIndexBatch, self._scalar_mix, self.bert_layers_mix_indices)
            emb = torch.cat([emb, packed_bert_encs.data],1)#  data,embed

        emb = torch.cat([word_fix_embed,emb],1)#  data,embed
 
        packed_emb = PackedSequence(emb, packed_input.batch_sizes)
        posterior_emb,amr_len = RelPosteriorUtils.posteriorIndictedEmb(packed_emb,packed_posterior)
        # padded_emb is padded tensor, (amr_len x batch) x len x n_feture, lengths
        padded_emb, lengths = unpack(posterior_emb, batch_first = True)
        max_len = padded_emb.size(1)
        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        if self.device_type == 'cuda':
            mask = mask.cuda()
        # logger.info("mask information is {}".format(mask))
        outputs = self.transformer_encoder(padded_emb, mask)
        # shape (batch_size, sequence_length, hidden_dim)
        packedOutputs = pack(outputs, lengths, batch_first = True)
        return  DoublePackedSequence(packedOutputs,amr_len,packedOutputs.data)

#multi pass sentence encoder for relation identification
class RelSentenceEncoder(nn.Module):

    def __init__(self, opt, embs, bert_model = None, bert_layers_mix_indices= None, char_encoder = None):
        super(RelSentenceEncoder, self).__init__()
        self.layers = opt.rel_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.txt_rnn_size % self.num_directions == 0
        self.hidden_size = opt.rel_rnn_size // self.num_directions
        self.rel_src_enc_size = opt.rel_rnn_size
        inputSize =  embs["word_fix_lut"].embedding_dim + embs["lemma_lut"].embedding_dim\
                     +embs["pos_lut"].embedding_dim+embs["ner_lut"].embedding_dim+1

        self.char_encoder = char_encoder
        # adding char
        if self.char_encoder:
            inputSize = inputSize + self.char_encoder.get_output_dim()
            self.char_lut = embs["char_lut"]

        # adding bert
        self.bert_model = bert_model
        if self.bert_model:
            self.mix_size  = len(bert_layers_mix_indices)
            self.bert_layers_mix_indices = bert_layers_mix_indices
            assert self.mix_size <= self.bert_model.config.num_hidden_layers and self.mix_size > 0, "wrong bert_layers_mix_indices = {}".format(bert_layer_mix_indices)
            if self.mix_size > 1:
                self._scalar_mix = ScalarMix(self.mix_size, do_layer_norm=False)
            else:
                self._scalar_mix = None
            inputSize = inputSize + self.bert_model.config.hidden_size

        self.rnn =nn.LSTM(inputSize, self.hidden_size,
                        num_layers=self.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn,
                           batch_first=True)   #first is for root

        self.lemma_lut = embs["lemma_lut"]
        self.word_fix_lut  = embs["word_fix_lut"]
        self.pos_lut = embs["pos_lut"]
        self.ner_lut = embs["ner_lut"]
        # self.alpha_dropout = nn.AlphaDropout(p = opt.alpha_dropout)
        self.alpha_dropout = opt.alpha_dropout
        if opt.cuda:
            self.rnn.cuda()

    def forward(self, packed_input, src_charBatch, packed_posterior,hidden=None,bertBatch=None, bertIndexBatch=None):
        assert isinstance(packed_input,PackedSequence)
        input = packed_input.data

        if self.alpha_dropout and self.training:
            input = data_dropout(input, self.alpha_dropout)
        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])

        emb = torch.cat([lemma_emb,pos_emb,ner_emb],1)#  data,embed
# add char
        if self.char_encoder:
            # inputs: (src_len x batch_size, n_characters, dim)
            # mask = (src_len x batch_size, n_characters)
            packed_char_encoding = CharEncoderUtils.encodeSequence(self.char_encoder, src_charBatch, self.char_lut)
            emb = torch.cat([emb,packed_char_encoding.data],1)#  data,embed

        # add bert
        if self.bert_model:
            packed_bert_encs,_ = BertEncoderUtils.encodeSequence(self.bert_model, packed_input, bertBatch, bertIndexBatch, self._scalar_mix, self.bert_layers_mix_indices)
            emb = torch.cat([emb, packed_bert_encs.data],1)#  data,embed

        emb = torch.cat([word_fix_embed,emb],1)#  data,embed
        emb = PackedSequence(emb, packed_input.batch_sizes)
        poster_emb,amr_len = RelPosteriorUtils.posteriorIndictedEmb(emb,packed_posterior)
        # poster_emb: packed(batch_size, )

        # Outputs : [packed(amr_len x batch_size), src_len, rnn_enc_size]
        Outputs = self.rnn(poster_emb, hidden)[0]

        return  DoublePackedSequence(Outputs,amr_len,Outputs.data)

class RelModel(nn.Module):
    def __init__(self, opt,embs, root_src_enc_size, rel_src_enc_size, frame="amr"):
        super(RelModel, self).__init__()
        if frame=="amr":
            self.root_encoder = AMRRootEncoder(opt,embs, root_src_enc_size)
            self.rel_encoder = AMRRelEncoder( opt, embs, rel_src_enc_size)
            self.rel_generator = RelCalssifierBiLinear( opt, embs,embs["amr_rel_lut"].num_embeddings)
        elif frame=="dm":
            self.root_encoder = DMRootEncoder(opt,embs, root_src_enc_size)
            self.rel_encoder = DMRelEncoder( opt, embs, rel_src_enc_size)
            self.rel_generator = RelCalssifierBiLinear( opt, embs,embs["dm_rel_lut"].num_embeddings)
        elif frame=="psd":
            self.root_encoder = PSDRootEncoder(opt,embs, root_src_enc_size)
            self.rel_encoder = PSDRelEncoder( opt, embs, rel_src_enc_size)
            self.rel_generator = RelCalssifierBiLinear( opt, embs,embs["psd_rel_lut"].num_embeddings)
        else:
            raise NotImplementedError("{} is not supported".format(frame))

        # root classifier also can be shared
        self.root = nn.Linear(opt.rel_dim,1)
        self.LogSoftmax = nn.LogSoftmax(dim=-1)


    def root_score(self,mypackedhead):
        heads = myunpack(*mypackedhead)
        output = []
        for head in heads:
            score = self.root(head).squeeze(1)
            output.append(self.LogSoftmax(score))
        return output

    def forward(self, srlBatch, index,src_enc,root_enc):
        # srlBatch: all_amr_nodes x n_feature,  AMR_CAT, AMR_LE, AMR_NER, AMR_SENSE, index of nodes,
        #  mypacked_seq[packed_batch_gold_amr_len x tgt_feature_dim]
        # index:  rel_index_batch:  list(batch, real_gold_amr_len), but content is the index of recatogrized amr index, is a mapping
        # src_enc: DoublePackedSequence(packed: packed_g_amr_len x re_amr_len x dim), length= re_lens)
        # root_enc: is weighted root_src_enc, mypack, list(batch, re_amr_len x txt_enc_size), re_amr_lens
        # mypacked_root_enc: MyPackedSequence(data: packed_gold_amr_len x dim, lengths: readl_gold_amr_len)
        # dim is opt.rel_dim
        mypacked_root_enc = self.root_encoder(srlBatch, index,root_enc) #with information from le cat enc
        roots = self.root_score(mypacked_root_enc)

        encoded= self.rel_encoder(srlBatch, index,src_enc)
        score_packed = self.rel_generator(*encoded)

        return score_packed,roots #,arg_logit_packed


class RelCalssifierBiLinear(nn.Module):
    """
    this code can be shared to use by different framework, except that n_rel should be specificed differently
    """
    def __init__(self, opt, embs,n_rel):
        super(RelCalssifierBiLinear, self).__init__()
        self.n_rel = n_rel
        self.inputSize = opt.rel_dim

        self.bilinear = nn.Sequential(nn.Dropout(opt.dropout),
                                  nn.Linear(self.inputSize,self.inputSize* self.n_rel))
        self.head_bias = nn.Sequential(nn.Dropout(opt.dropout),
                                   nn.Linear(self.inputSize,self.n_rel))
        self.dep_bias = nn.Sequential(nn.Dropout(opt.dropout),
                                      nn.Linear(self.inputSize,self.n_rel))
        #self.bias = nn.Parameter(torch.normal(torch.zeros(self.n_rel)).cuda())

        self.bias = nn.Parameter(torch.normal(torch.zeros(self.n_rel)))

        if opt.cuda:
            self.cuda()

    def bilinearForParallel(self,inputs,length_pairs):
        output = []
        ls = []
        for i,input in enumerate(inputs):

            #head_t : amr_l x (  rel_dim x n_rel)
            #dep_t : amr_l x amr_l x rel_dim
            #head_bias : amr_l  x n_rel
            #dep_bias : amr_l  x   amr_l  x n_rel
            head_t,dep_t,head_bias,dep_bias = input
            l = len(head_t)
            ls.append(l)
            head_t = head_t.view(l,-1,self.n_rel)
            score =dep_t[:,:length_pairs[i][1]].bmm( head_t.view(l,-1,self.n_rel)).view(l,l,self.n_rel).transpose(0,1)

            dep_bias =  dep_bias[:,:length_pairs[i][1]]
            score = score + dep_bias

            score = score + head_bias.unsqueeze(1).expand_as(score)
            score = score+self.bias.unsqueeze(0).unsqueeze(1).expand_as(score)
            score = F.log_softmax(score.view(ls[-1]*ls[-1],self.n_rel), dim=-1) # - score.exp().sum(2,keepdim=True).log().expand_as(score)
            
            output.append(score.view(ls[-1]*ls[-1],self.n_rel))
        return output,[l**2 for l in ls]


    def forward(self, _,heads,deps):
        '''heads.data: mypacked        amr_l x rel_dim
            deps.data: mydoublepacked     amr_l x amr_l x rel_dim
        '''
        heads_data = heads.data
        deps_data = deps.data

        head_bilinear_transformed = self.bilinear (heads_data)  #all_data x (    n_rel x inputsize)

        head_bias_unpacked = myunpack(self.head_bias(heads_data),heads.lengths) #[len x n_rel]

        size = deps_data.size()
        dep_bias =  self.dep_bias(deps_data.view(-1,size[-1])).view(size[0],size[1],-1)

        dep_bias_unpacked,length_pairs = mydoubleunpack(MyDoublePackedSequence(MyPackedSequence( dep_bias,deps[0][1]),deps[1],dep_bias) ) #[len x n_rel]

        bilinear_unpacked = myunpack(head_bilinear_transformed,heads.lengths)

        deps_unpacked,length_pairs = mydoubleunpack(deps)
        output,l = self.bilinearForParallel( zip(bilinear_unpacked,deps_unpacked,head_bias_unpacked,dep_bias_unpacked),length_pairs)
        myscore_packed = mypack(output,l)

      #  prob_packed = MyPackedSequence(myscore_packed.data,l)
        return myscore_packed
