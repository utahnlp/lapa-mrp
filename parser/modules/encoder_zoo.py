#!/usr/bin/env python3.6
# coding=utf-8
'''
Encoder zoo for creating encoder modules from config
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2018-05-30
'''

import torch
import torch.nn as nn
from parser.modules.helper_module import *
from parser.modules.bert_utils import *
from parser.modules.char_utils import *
from pytorch_transformers.modeling_bert import *
from allennlp.modules.seq2seq_encoders.stacked_self_attention import *
from torch.nn.utils.rnn import PackedSequence
from utility.constants import *
from pytorch_transformers.modeling_bert import BertModel

import json
import logging
logger = logging.getLogger("encoder_zoo")

class EncoderZoo:
    @staticmethod
    def create_transformer_encoder_from_json_config(transformer_json_config, opt):
        """
        transformer configuration:
        "hidden_dim:projection_dim:feedforward_hidden_dim:num_layers:num_atttention_heads:use_positional_encoding:dropout_prob:residual_dropout_prob:attention_dropout_prob"
        """
        if isinstance(transformer_json_config, str):
            logger.info("transformer_config{}".format(transformer_json_config))
            transformer_config = json.loads(transformer_json_config)
            logger.info("load transformer_config: {}".format(transformer_config))
        else:
            # when it is a json object or other, just assign it
            transformer_config= transformer_json_config

        assert transformer_config, "transformer_config:{}".format(transformer_json_config)
        # required paramters for stacked_self_attention
        input_dim = transformer_config['input_dim']
        hidden_dim = transformer_config['hidden_dim']
        projection_dim = transformer_config['projection_dim']
        feedforward_hidden_dim = transformer_config['feedforward_hidden_dim']
        num_layers = transformer_config['num_layers']
        num_attention_heads = transformer_config['num_attention_heads']
        use_postitional_encoding = transformer_config.get('use_positinal_encoding', True)
        dropout_prob = transformer_config.get('dropout_porb', opt.dropout)
        residual_dropout_prob = transformer_config.get('residual_dropout_prob', opt.dropout)
        attention_dropout_prob = transformer_config.get('attention_dropout_prob', opt.dropout)

        transformer_encoder = StackedSelfAttentionEncoder(input_dim, hidden_dim, projection_dim,
                                                          feedforward_hidden_dim, num_layers,
                                                          num_attention_heads, use_postitional_encoding,
                                                          dropout_prob, residual_dropout_prob,
                                                          attention_dropout_prob)
        return transformer_encoder

    @staticmethod
    def create_bert_model_from_json_config(bert_json_config, opt):
        if bert_json_config:
            logger.info("bert_json_config{}".format(bert_json_config))
            bert_config = json.loads(bert_json_config)
            logger.info("load bert_json_config: {}".format(bert_config))
        else:
            bert_config= None

        assert bert_config, "bert_config:{}".format(bert_json_config)
        # use seperate bert model
        config = BertConfig.from_pretrained(bert_config['bert_config_name'])
        config.output_attentions = True
        config.output_hidden_states = True
        bert_model = BertModel(config)
        return bert_model

    @staticmethod
    def create_char_encoder_from_json_config(char_json_config, opt):
        if char_json_config:
            logger.info("char_json_config{}".format(char_json_config))
            char_config = json.loads(char_json_config)
            logger.info("load char_json_config: {}".format(char_config))
        else:
            char_config= None

        assert char_config, "char_config:{}".format(char_config)
        char_encoder_type = char_config['char_encoder_type']
        if char_encoder_type == "CnnHighway":
            char_embedding_dim = char_config['char_emb_dim'] if 'char_emb_dim' in char_config else opt.char_dim
            filters = char_config['filters'] if 'filters' in char_config else [[2, 8], [3, 16], [4, 32], [5, 64]]
            num_high_way= char_config['num_high_way'] if 'num_high_way' in char_config else 2
            projection_dim= char_config['projection_dim'] if 'projection_dim' in char_config else 100
            projection_location= char_config['projection_location'] if 'projection_location' in char_config else 'after_cnn'
            do_layer_norm=False
            char_encoder = CnnHighwayEncoder(
                activation='relu',
                embedding_dim=char_embedding_dim,
                filters = filters,
                num_highway=2,
                projection_dim=projection_dim,
                projection_location='after_highway',
                do_layer_norm=do_layer_norm
            )
        elif char_encoder_type == "Cnn":
            char_encoder = CnnEncoder(embedding_dim=64,
                                      num_filters=25,
                                      ngram_filter_sizes=(2, 3, 4, 5))
        else:
            raise NotImplementedError("{} is not supported".format(char_encoder_type))
            # use seperate bert model
        return char_encoder

    @staticmethod
    def create_sentence_encoder_from_config(embs, component_dict, snt_encoder_config, opt):
        """
        CONCEPT_SNT_ENCODER=bert:rel:s:12:{\"bert_config_name\":\"bert-base-cased\"}¬
        """
        snt_encoder = None
        if 'char_encoder' in component_dict:
            char_encoder = component_dict['char_encoder']
        else:
            char_encoder = None
        encode_configs = snt_encoder_config.split(":")
        component_type = encode_configs[0]
        if len(encode_configs) > 1:
            component_id = encode_configs[1]
        if component_type == 'rnn':
            # Now rnn is not shared
            snt_encoder = SentenceEncoder( opt, embs, src_enc_size = opt.txt_rnn_size, char_encoder = char_encoder)
        elif component_type == 'transformer':
            if component_id in component_dict:
                transformer_encoder = component_dict[component_id]
            else:
                remaining_concept_json_configs = ':'.join(encode_configs[2:])
                transformer_encoder = EcnoderZoo.create_transformer_encoder_from_json_config(remaining_concept_json_configs, opt)
                if opt.cuda > 0:
                    if opt.cuda == 2:
                        device_name = 'cuda:1'
                    else:
                        device_name = 'cuda:{}'.format((len(component_dict)+1)%opt.cuda)
                else:
                    device_name = 'cpu'
                transformer_encoder.to(torch.device(device_name))
                logger.info("{} initialized on device {}".format(opt.concept_snt_encoder, device_name))
                component_dict[component_id] = transformer_encoder
            snt_encoder = TransformerSentenceEncoder(opt, embs, transformer_encoder, char_encoder = char_encoder)
        elif component_type.startswith('bert'):
            # for selected layers
            mode = encode_configs[2]
            selected_layers_str = encode_configs[3]
            if ',' in selected_layers_str:
                layers_indices = [int(i_str) for i_str in selected_layers_str.split(',') if i_str]
            elif '-' in selected_layers_str:
                split_strs = selected_layers_str.split('-')
                assert len(split_strs) ==2, "range of layers must be like 0-6, 2-5, close set"
                start = int(split_strs[0])
                end = int(split_strs[1])
                layers_indices = range(start, end+1)
            else:
                # top layers, -N: 0
                layers_indices = range(-int(selected_layers_str), 0)

            if component_id not in component_dict:
                # use seperate bert model
                bert_json_config = ':'.join(encode_configs[4:])
                sep_bert_model = EncoderZoo.create_bert_model_from_json_config(bert_json_config, opt)
                if opt.cuda > 0:
                    device_name = 'cuda:{}'.format((len(component_dict)+1)%opt.cuda)
                else:
                    device_name = 'cpu'

                sep_bert_model.to(torch.device(device_name))
                logger.info("{} initialized on device {}".format(opt.concept_snt_encoder, device_name))
                if mode == 'd':
                    sep_bert_model.train()
                elif mode == 's':
                    sep_bert_model.eval()
                else:
                    raise RuntimeError("Wrong configuration for mode of bert model {}".format(bert_json_config))
                # record this newly created bert_model
                component_dict[component_id] = sep_bert_model
                if component_type == "bert":
                    snt_encoder = BertSentenceEncoder(opt, sep_bert_model, layers_indices, src_enc_size = sep_bert_model.config.hidden_size)
                elif component_type == "bert_rnn":
                    snt_encoder = SentenceEncoder(opt, embs, opt.txt_rnn_size, sep_bert_model, layers_indices, char_encoder = char_encoder)
                elif component_type == "bert_rnn_char":
                    char_encoder = xx
                    snt_encoder = SentenceEncoder(opt, embs, opt.txt_rnn_size, sep_bert_model, layers_indices, char_encoder = char_encoder)
                else:
                    raise NotImplementedError("{} is not implemented".format(component_type))
            else:
                # use existed bert model, make sure the share model used in same mode
                existed_bert_model = component_dict[component_id]
                assert (mode == 'd' and existed_bert_model.training == True) or (mode == 's' and existed_bert_model.training == False), "shared bert must be used consistently for concept_snt_encoder : id = {}, training:{}".format(component_id, existed_bert_model.training)
                if component_type == "bert":
                    snt_encoder = BertSentenceEncoder(opt, existed_bert_model, layers_indices, src_enc_size = existed_bert_model.config.hidden_size)
                elif component_type == "bert_rnn":
                    snt_encoder = SentenceEncoder(opt, embs, opt.txt_rnn_size, existed_bert_model, layers_indices, char_encoder = char_encoder)
                else:
                    raise NotImplementedError("{} is not implemented".format(component_type))
        else:
            raise NotImplementedError("{} is not supported".format(opt.concept_snt_encoder))

        return snt_encoder

class BertSentenceEncoder(nn.Module):
    def __init__(self, opt, bert_model, bert_layers_mix_indices, src_enc_size):
        # for a single sentence encoder, type id are full of zeors
        # input mask need to be extened with lengths.
       # https://github.com/allenai/allennlp/blob/e9287d4d48d2d980226a4acf70d03eedda67c548/allennlp/modules/scalar_mix.py
        super(BertSentenceEncoder, self).__init__()
        self.opt = opt
        self.src_enc_size = src_enc_size
        self.bert_model = bert_model
        self.mix_size  = len(bert_layers_mix_indices)
        self.bert_layers_mix_indices = bert_layers_mix_indices
        assert self.mix_size <= self.bert_model.config.num_hidden_layers and self.mix_size > 0, "wrong bert_layers_mix_indices = {}".format(bert_layer_mix_indices)
        if self.mix_size > 1:
            self._scalar_mix = ScalarMix(self.mix_size, do_layer_norm=False)
        else:
            self._scalar_mix = None

    def forward(self, packed_input: PackedSequence, src_charBatch, bertBatch = None, bertIndexBatch=None):
        if packed_input.data.size(-1) == self.src_enc_size:
            # logger.info("Just use packed input as output:{}".format(packed_input))
            # it is already encoded
            return packed_input
        packed_bert_encs,_ = BertEncoderUtils.encodeSequence(self.bert_model, packed_input, bertBatch, bertIndexBatch, self._scalar_mix, self.bert_layers_mix_indices)
        return packed_bert_encs

class TransformerSentenceEncoder(nn.Module):
    """
    take sentence as input, output a encoding for each token in it.
    """
    def __init__(self, opt, embs, transformer_encoder, bert_model = None, bert_layers_mix_indices= None, char_encoder = None):
        super(TransformerSentenceEncoder, self).__init__()
        assert transformer_encoder,"transformer_encoder is None"
        self.transformer_encoder = transformer_encoder
        self.src_enc_size = transformer_encoder.get_output_dim()
        inputSize = embs["word_fix_lut"].embedding_dim +  embs["lemma_lut"].embedding_dim\
                    +embs["pos_lut"].embedding_dim + embs["ner_lut"].embedding_dim

        self.char_encoder = char_encoder
        if self.char_encoder:
            inputSize = input_Size + self.char_encoder.get_output_dim()
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

        # https://github.com/allenai/allennlp/blob/master/allennlp/modules/seq2seq_encoders/stacked_self_attention.py
        assert self.transformer_encoder.get_input_dim() == inputSize, "inputSize not match {} != actual encoder_size{}".format(self.transformer_encoder.get_input_dim(), inputSize)
        self.lemma_lut = embs["lemma_lut"]
        self.word_fix_lut  = embs["word_fix_lut"]
        self.pos_lut = embs["pos_lut"]
        self.ner_lut = embs["ner_lut"]
        # use the same dropout parameters for all the dropout layer.
        self.drop_emb = nn.Dropout(opt.dropout)
        # self.alpha_dropout = nn.AlphaDropout(p=opt.alpha_dropout)
        self.alpha_dropout = opt.alpha_dropout

        if opt.cuda:
            self.device_type = 'cuda'
            self.cuda()

    def forward(self, packed_input: PackedSequence, src_charBatch, hidden=None, bertBatch=None, bertIndexBatch=None):
        # input: pack([src_len x batch_size, n_feature], batch_size)
        # https://stackoverflow.com/questions/37720869/emacs-how-do-i-set-flycheck-to-python-3
        # https://pytorch.org/docs/stable/nn.html?highlight=packed#torch.nn.utils.rnn.PackedSequence
        # Here, input is the data in the packedSequence, not a padded sequence
        if packed_input.data.size(-1) == self.src_enc_size:
            #logger.info("Just use packed input as output:{}".format(packed_input))
            # it is already encoded
            return packed_input
        input = packed_input.data
        if self.alpha_dropout and self.training:
            input = data_dropout(input, self.alpha_dropout)

        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])

        # it only dropout the other embeb, for word embed , it did not dropout
        emb = self.drop_emb(torch.cat([lemma_emb,pos_emb,ner_emb],1))#  data,embed
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
        # emb : (src_len x batch_size, n_dim), packed_batch_size
        packed_emb =  PackedSequence(emb, packed_input.batch_sizes)
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


class SentenceEncoder(nn.Module):
    """
    take sentence as input, output a encoding for each token in it.
    """
    def __init__(self, opt, embs, src_enc_size, bert_model = None, bert_layers_mix_indices= None, char_encoder = None):
        super(SentenceEncoder, self).__init__()
        self.layers = opt.txt_enlayers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.txt_rnn_size % self.num_directions == 0
        self.hidden_size = opt.txt_rnn_size // self.num_directions
        self.src_enc_size = src_enc_size
        assert self.src_enc_size == opt.txt_rnn_size, "src_enc_size:{} is not txt_rnn_size:{}".format(src_enc_size, txt_rnn_size)

        self.char_encoder = char_encoder
    #    inputSize = opt.word_dim*2 + opt.lemma_dim + opt.pos_dim +opt.ner_dim
        inputSize = embs["word_fix_lut"].embedding_dim +  embs["lemma_lut"].embedding_dim\
                    +embs["pos_lut"].embedding_dim + embs["ner_lut"].embedding_dim

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

        # it is not batch first,
        self.rnn = nn.LSTM(inputSize, self.hidden_size,
                        num_layers=self.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)

        self.lemma_lut = embs["lemma_lut"]
        self.word_fix_lut  = embs["word_fix_lut"]
        self.pos_lut = embs["pos_lut"]
        self.ner_lut = embs["ner_lut"]
        # use the same dropout parameters for all the dropout layer.
        self.drop_emb = nn.Dropout(opt.dropout)
        # self.alpha_dropout = nn.AlphaDropout(p=opt.alpha_dropout)
        self.alpha_dropout = opt.alpha_dropout

        if opt.cuda:
            self.rnn.cuda()


    def forward(self, packed_input: PackedSequence, src_charBatch, hidden=None, bertBatch=None, bertIndexBatch=None):
        # input: pack([src_len x batch_size, n_feature], batch_size)
        # https://stackoverflow.com/questions/37720869/emacs-how-do-i-set-flycheck-to-python-3
        # https://pytorch.org/docs/stable/nn.html?highlight=packed#torch.nn.utils.rnn.PackedSequence
        # Here, input is the data in the packedSequence, not a padded sequence
        if packed_input.data.size(-1) == self.src_enc_size:
            #logger.info("Just use packed input as output:{}".format(packed_input))
            # it is already encoded
            return packed_input
        input = packed_input.data
        if self.alpha_dropout and self.training:
            input = data_dropout(input, self.alpha_dropout)

        # pack(src_len x batch_size, dim)
        word_fix_embed = self.word_fix_lut(input[:,TXT_WORD])
        lemma_emb = self.lemma_lut(input[:,TXT_LEMMA])
        pos_emb = self.pos_lut(input[:,TXT_POS])
        ner_emb = self.ner_lut(input[:,TXT_NER])
        emb = self.drop_emb(torch.cat([lemma_emb,pos_emb,ner_emb],1))#  data,embed
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
        # emb : (src_len x batch_size, n_dim), packed_batch_size
        emb =  PackedSequence(emb, packed_input.batch_sizes)
        # https://pytorch.org/docs/stable/nn.html?highlight=rnn#torch.nn.RNN
        # if a packed sequence is given, then output is also a packed sequence.
        # [seq_len x batch, num_directions*hidden_size], packed_input.batch_sizes
        outputs, hidden_t = self.rnn(emb, hidden)
        return  outputs
