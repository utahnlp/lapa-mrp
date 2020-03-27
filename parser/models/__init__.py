#!/usr/bin/env python3.6
# coding=utf-8
'''

Deep Learning Models for variational inference of alignment.
Posterior , LikeliHood helps computing posterior weighted likelihood regarding relaxation.

Also the whole AMR model is combined here.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import numpy as np
from parser.models.ConceptModel import *
from parser.models.MultiPassRelModel import *
from parser.models.AMRGraphModel import *
from parser.models.VariationalModel import *
from parser.modules.encoder_zoo import *

from parser.modules.GumbelSoftMax import renormalize,sink_horn,gumbel_noise_sample
from parser.modules.helper_module import *

from pytorch_transformers.modeling_bert import BertModel
from allennlp.modules.seq2seq_encoders.stacked_self_attention import *
import json

from copy import deepcopy
import logging
logger = logging.getLogger("mrp")
class MTLModel(nn.Module):
    def __init__(self, model_dicts):
        super(MTLModel, self).__init__()
        self.amr_model = model_dicts["amr"] if "amr" in model_dicts else None
        self.dm_model = model_dicts["dm"] if "dm" in model_dicts else None
        self.psd_model = model_dicts["psd"] if "psd" in model_dicts else None
        self.eds_model = model_dicts["eds"] if "eds" in model_dicts else None
        self.ucca_model = model_dicts["ucca"] if "ucca" in model_dicts else None

    def forward(self):
        # pass
        return

#The entire model assembled
class MRPModel(nn.Module):
    # bert_model is unique global instance
    def __init__(self,opt,embs, component_dict=None, frame = "amr"):
        super(MRPModel, self).__init__()
        self.component_dict = component_dict
        if opt.char_encoder_config:
            self.component_dict['char_encoder'] = EncoderZoo.create_char_encoder_from_json_config(opt.char_encoder_config, opt)
        self.frame = frame
        self.concept_decoder = ConceptIdentifier(opt, embs, encoder = None, component_dict = self.component_dict, frame=self.frame)
        self.posterior_m = VariationalAlignmentModel(opt, embs, encoder = None, component_dict = self.component_dict, frame=self.frame)
        self.opt = opt
        self.embs = embs
        self.rel = opt.rel
        if opt.rel :
            self.start_rel(opt)
        if opt.cuda:
            self.cuda()

    def init_posterior_m(self, opt, embs):
        self.independent_posterior = opt.independent_posterior
        # when using embedding features
        if opt.emb_independent:
            to_be_copied = ["lemma_lut","pos_lut","ner_lut"]
            for n in to_be_copied:
                embs[n] = deepcopy(self.embs[n])
                embs[n].requires_grad = True

            if self.frame == "amr":
                embs["amr_cat_lut"] = deepcopy(self.embs["amr_cat_lut"])
                embs["amr_cat_lut"].requires_grad = True
                self.posterior_m.amr_encoder.amr_cat_lut = embs["amr_cat_lut"]
                self.posterior_m.amr_encoder.lemma_lut = embs["lemma_lut"]
            # if share the encoder
            if not self.independent_posterior :
                self.posterior_m.snt_encoder  = deepcopy(self.concept_decoder.snt_encoder)

            # copy emdedding only when not using bert
            if isinstance(self.posterior_m.snt_encoder, SentenceEncoder):
                self.posterior_m.snt_encoder.pos_lut = embs["pos_lut"]
                self.posterior_m.snt_encoder.lemma_lut = embs["lemma_lut"]
                self.posterior_m.snt_encoder.ner_lut = embs["ner_lut"]

    def init_rel_src_encoder(self, opt, embs):
        if 'char_encoder' in self.component_dict:
            char_encoder = self.component_dict['char_encoder']
        else:
            char_encoder = None
        # rel_src_encoder
        rel_src_encode_configs = opt.rel_snt_encoder.split(":")
        rel_src_component_type = rel_src_encode_configs[0]
        if len(rel_src_encode_configs) > 1:
            rel_src_component_id = rel_src_encode_configs[1]
        if rel_src_component_type == 'rnn':
            # Now rnn is not shared
            self.rel_src_encoder = RelSentenceEncoder(opt, embs, char_encoder = char_encoder)
            self.rel_src_enc_size = self.rel_src_encoder.rel_src_enc_size
        elif rel_src_component_type == 'transformer':
            if rel_src_component_id in self.component_dict:
                transformer_encoder = self.component_dict[rel_src_component_id]
            else:
                remaining_rel_json_configs = ':'.join(rel_src_encode_configs[2:])
                transformer_encoder = EncoderZoo.create_transformer_encoder_from_json_config(remaining_rel_json_configs, opt)
                if opt.cuda > 0:
                    device_name = 'cuda:{}'.format((len(self.component_dict)+1)%opt.cuda)
                else:
                    device_name = 'cpu'
                transformer_encoder.to(torch.device(device_name))
                logger.info("{} initialized on device {}".format(opt.rel_snt_encoder, device_name))
                self.component_dict[rel_src_component_id] = transformer_encoder
            self.rel_src_encoder = RelTransformerSentenceEncoder(opt, embs, transformer_encoder, char_encoder= char_encoder)
            self.rel_src_enc_size = self.rel_src_encoder.rel_src_enc_size
        elif rel_src_component_type.startswith('bert'):
            # use bert model
            mode = rel_src_encode_configs[2]
            selected_layers_str = rel_src_encode_configs[3]
            if ',' in selected_layers_str:
                layers_indices = [int(i_str) for i_str in selected_layers_str.split(',') if i_str]
            elif '-' in selected_layers_str:
                split_strs = selected_layers_str.split('-')
                assert len(split_strs) ==2, "range of layers must be like 0-12, 2-5, close set"
                start = int(split_strs[0])
                end = int(split_strs[1])
                layers_indices = range(start, end+1)
            else:
                # top layers, -N: 0
                layers_indices = range(-int(selected_layers_str), 0)

            if rel_src_component_id not in self.component_dict:
                # use seperate bert model
                bert_json_config = ':'.join(rel_src_encode_configs[4:])
                sep_bert_model = EncoderZoo.create_bert_encoder_from_json_config(bert_json_config,opt)
                if opt.cuda > 0:
                    if opt.cuda == 2:
                        device_name = 'cuda:1'
                    else:
                        device_name = 'cuda:{}'.format((len(self.component_dict)+1)%opt.cuda)
                else:
                    device_name = 'cpu'
                sep_bert_model.to(torch.device(device_name))
                logger.info("{} initialized on device {}".format(opt.rel_snt_encoder, device_name))
                if mode == 'd':
                    sep_bert_model.train()
                elif mode == 's':
                    sep_bert_model.eval()
                else:
                    raise RuntimeError("Wrong configuration for mode of bert model {}".format(bert_json_config))
                # record this newly created bert_model
                self.component_dict[rel_src_component_id] = sep_bert_model
                if rel_src_component_type == "bert":
                    self.rel_src_encoder = RelBertSentenceEncoder(opt, sep_bert_model, layers_indices)
                elif rel_src_component_type == "bert_rnn":
                    self.rel_src_encoder = RelSentenceEncoder(opt, embs, bert_model = sep_bert_model, bert_layers_mix_indices = layers_indices, char_encoder = char_encoder)
                else:
                    raise NotImplementedError("{} is not implemented".format(rel_src_component_type))

                self.rel_src_enc_size = self.rel_src_encoder.rel_src_enc_size
            else:
                # use shared bert model
                existed_bert_model = self.component_dict[rel_src_component_id]
                assert (mode == 'd' and existed_bert_model.training == True) or (mode == 's' and existed_bert_model.training == False), "shared bert must be used consistently for rel_snt_encoder: id = {}, training:{}".format(rel_src_component_id, existed_bert_model.training)
                if rel_src_component_type == "bert":
                    self.rel_src_encoder = RelBertSentenceEncoder(opt, existed_bert_model, layers_indices)
                elif rel_src_component_type == "bert_rnn":
                    self.rel_src_encoder = RelSentenceEncoder(opt,embs, existed_bert_model, layers_indices, char_encoder = char_encoder)
                else:
                    raise NotImplementedError("{} is not implemented".format(rel_src_component_type))
                self.rel_src_enc_size = self.rel_src_encoder.rel_src_enc_size
        else:
            raise NotImplementedError("rel_snt_encoder {} is not supported".format(opt.rel_snt_encoder))


    def init_root_src_encoder(self, opt, embs):
        """:
        init root_src_encoder
        """
        self.root_src_encoder = EncoderZoo.create_sentence_encoder_from_config(embs, self.component_dict, opt.root_snt_encoder, opt)
        self.root_src_enc_size = self.root_src_encoder.src_enc_size

    def start_rel(self,opt):
    #        return
        self.rel = True
        size = opt.txt_rnn_size
        assert int(size/2) == size/2
        embs = self.embs

        if self.posterior_m:
            self.init_posterior_m(opt, embs)

        if opt.emb_independent:
            to_be_copied = ["lemma_lut","pos_lut","ner_lut", "amr_cat_lut"] if self.frame == "amr" else ["lemma_lut","pos_lut","ner_lut"]
            for n in to_be_copied:
                embs[n] = deepcopy(self.embs[n])
                embs[n].requires_grad = True

        self.init_rel_src_encoder(opt, embs)
        self.init_root_src_encoder(opt, embs)
        self.relModel = RelModel(opt, embs, self.root_src_enc_size, self.rel_src_enc_size, frame=self.frame)

#srl index is from original node to recategorized, combine with posterior of recategorized concepts
    #we get the posterior needed for relation prediction
    def index_posterior(self,posterior,rel_index):
        '''rel_index:  # batch x var(gold_amr_len), every index is the index in re_amr_len
            posterior :  packed(packed_batch_re_amr_len x  src_len), re_amr_len

            out:    batch x var(gold_amr_len  x src_len ), gold_amr_l
            '''
        # posterior : padded_re_amr_len x batch x src_len
        posterior,lengths = unpack(posterior)

        # out_l, the length of gold amr nodes, the index is to the original categorized amr
        out_l = [len(i) for i in rel_index]
        out_posterior = []
        for i,l in enumerate(out_l):
            # i is the ith sentence,
            # l is the length for recategorized amr, which is the index in posterior
            # select the posterior corresponding the every node in the output AMR
            # list(batch_size, real_gold_amr_len x src_len)
            # select all the original snt index for each current node index(after expansion)
            out_posterior.append(posterior[:,i,:].index_select(0,rel_index[i]))

        return mypack(out_posterior,out_l)

    def forward(self, input,rel=False, bertBatch=None,bertIndexBatch=None):
        if len(input)==4 and not rel:
            # concept only
            # srcBatch: packed[packed_batch_src_len x src_feature_dim]
            # tgtBatch: packed[packed_batch_re_amr_len x tgt_feature_dim]
            # alginbatch: packed[packed_batch_re_amr_len x src_len]
            # srcBertBatch: tensor [batch x bert_max_length]
            # probBatch: PackedSequence(cat_prob,batch_sizes),PackedSequence(le_prob,batch_sizes),PackedSequence(ner_prob,batch_sizes), all is packed_src_len x dim
            # posterior_likelihoood_score: posterior, likelihood, score
            # posterior: packed_batch_re_amr_len x src_len, batch_sizes
            # likelihood: packed_batch_re_amr_len x src_len, batch_sizes
            # score: packed_batch_re_amr_len x src_len, batch_sizes
            #training concept identification, alignBatch here indicating possibility of copying (or aligned by rule)
            srcBatch,src_charBatch, tgtBatch,alginBatch = input
            # return (cat_prob, batch_sizes), (le_prob, batch_size), (ner_prob, batch_size), src_enc: (src_len_batch,rnn_size)
            probBatch,src_enc = self.concept_decoder(srcBatch, src_charBatch, bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)
            # logger.info("Concept_decoder: finished: probBatch:{}, src_enc:{}".format(probBatch, src_enc))

            if self.opt.use_src_encs_for_posterior:
                posteriors_likelihood_score = self.posterior_m(src_enc,tgtBatch,alginBatch,probBatch,bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)
            else:
                posteriors_likelihood_score = self.posterior_m(srcBatch,src_charBatch, tgtBatch,alginBatch,probBatch,bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)
            # logger.info("posterior :{}, linkhood:{}, scores:{} ".format(*posteriors_likelihood_score))
            return probBatch,posteriors_likelihood_score,src_enc


        if len(input)==6 and rel:
            # rel_batch:  mypacked_seq[packed_batch_gold_amr_len x tgt_feature_dim]
            # rel_index_batch:  list(batch, real_gold_amr_len), but content is the index of recatogrized amr index, is a mapping
            # srcBatch: packed[packed_batch_src_len x src_feature_dim]
            # srcEnc : packed [packed_batch_src_len x src_enc_dim]
            # posterior: packed [packed_batch_re_amr_len x src_len, batch_sizes]
            # srcBertBatch: tensor [batch x bert_max_length]
            #training relation identification
            rel_batch,rel_index,srcBatch, src_charBatch, src_enc, posterior = input
            assert not np.isnan(np.sum(posterior.data.detach().cpu().numpy())),("posterior.data \n",posterior.data)
            posterior_data = renormalize(posterior.data+epsilon)
            assert not np.isnan(np.sum(posterior_data.detach().cpu().numpy())),("posterior_data \n",posterior_data)
            # normalized posterior pack(data(amr))
            # re_amr_len_batch x src_len
            posterior = PackedSequence(posterior_data,posterior.batch_sizes)
            # here it get all the possible alignment for each original gold node, by using a mapping from gold node amr to recategorized id
            # mypack(packed_gold_amr_len  x src_len), gold_amr_l
            indexed_posterior = self.index_posterior(posterior,rel_index)
            # logger.info("indexed_posterior:{}".format(indexed_posterior))

            # src_enc,DoublePack(outputs, amr_len, outputs.data)
            # outputs: mypack(data,lengths,batch_first=True): [packed_real_gold_amr_len x src_len, dim+1], src_lens
            # it take all packed_real_gold_amr_len as batch_size, it means for every gold amr node in every batch, it has a src_len sequence source encoding for it.
            if self.opt.use_src_encs_for_rel:
                rel_src_enc = self.rel_src_encoder(src_enc,src_charBatch,indexed_posterior, bertBatch=bertBatch, bertIndexBatch = bertIndexBatch)
            else:
                rel_src_enc = self.rel_src_encoder(srcBatch,src_charBatch,indexed_posterior, bertBatch=bertBatch, bertIndexBatch = bertIndexBatch)

            if self.opt.use_src_encs_for_root:
                # pack_src_len x dim
                root_src_enc =  self.root_src_encoder(src_enc, src_charBatch, bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)
            else:
                root_src_enc =  self.root_src_encoder(srcBatch, src_charBatch,bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)

            # logger.info("rel_src_enc:{}, root_src_enc ={}".format(rel_src_enc, root_src_enc))
            # weighted_root_enc: mypack, list(batch, max_re_amr_len x txt_enc_size), max_amr_lens
            weighted_root_src_enc = self.root_posterior_enc(posterior,root_src_enc)
            # weighted_enc: doublepack(list(batch, re_amr_len x g_amr_len x dim), list(g_amr_l, src_l))
            weighted_rel_src_enc= self.weight_posterior_enc(posterior,rel_src_enc)   #src_enc MyDoublePackedSequence, amr_len
            # logger.info("weighted_rel_src_enc:{}, weighted_root_src_enc ={}".format(weighted_rel_src_enc, weighted_root_src_enc))
            rel_prob = self.relModel(rel_batch,rel_index,weighted_rel_src_enc,weighted_root_src_enc)

            return rel_prob

        # evaluation on relation.
        if len(input)==5 and rel:
            # relation identification evaluation
            rel_batch,srcBatch, src_charBatch, src_enc, alginBatch = input   #
            # here alignBatch is some prior alignments, called real alignment, because only concepts can be aligned. When training, it is posterior alignments.
            if self.opt.use_src_encs_for_rel:
                rel_src_enc = self.rel_src_encoder(src_enc,src_charBatch ,alginBatch, bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)
            else:
                rel_src_enc = self.rel_src_encoder(srcBatch,src_charBatch, alginBatch, bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)

            if self.opt.use_src_encs_for_root:
                root_src_enc = self.root_src_encoder(src_enc, src_charBatch, bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)
            else:
                root_src_enc = self.root_src_encoder(srcBatch, src_charBatch, bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)

            root_data,lengths = unpack(root_src_enc)
            mypacked_root_enc = mypack(root_data.transpose(0,1).contiguous(),lengths)
            rel_prob = self.relModel(rel_batch,alginBatch,rel_src_enc,mypacked_root_enc)
            return rel_prob
        else:
        # concept identification evaluation
            srcBatch, src_charBatch = input
            probBatch,src_enc= self.concept_decoder(srcBatch, src_charBatch, bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)
        return probBatch, src_enc


    #encoding relaxation for root identification
    def root_posterior_enc(self,posterior,src_enc):
        '''src_enc:  # pack(src_l) x dim
               posterior =  re_amr_len x  batch x src_len , re_amr_l

               out: list(batch, re_amr_len x txt_enc_size), amr_lens
        '''
        # [max_re_amr_len, batch , src_len]
        posterior,lengths = unpack(posterior)
        # [max_src_len, batch, src_enc_dim]
        enc,length_src = unpack(src_enc)
        weighted = []
        for i, src_l in enumerate(length_src): #src_len  x dim
            # p: [amr_len, src_l]
            p = posterior[:,i,:src_l] #max_re_amr_len  x real_src_len
            # enc_t :[src_l, src_enc_dim]
            enc_t = enc[:src_l,i,:] # [real_src_len,src_enc_dim]
            # weighted_enc: amr_len x src_enc_dim
            weighted_enc = p.mm(enc_t)   #max_re_amr_len x src_enc_dim
            weighted.append(weighted_enc)  #max_re_amr_len x src_enc_dim
            # make a padded list into a pack
        return mypack(weighted,lengths)

    #encoding relaxation for relation identification
    def weight_posterior_enc(self,posterior,src_enc):
        '''
        src_enc : DoublePackedSequence:  [packed(pack_gold_amr_len x src_len x (src_enc_dim +1)), amr_lens, packed.data], 1 is indicator
        posterior =  re_amr_len x  batch x src_len , re_amr_l
        out: batch x amr_len x txt_enc_size
        '''
        # lengths , re_amr_len for each batch
        posterior,lengths = unpack(posterior)
        # posterior : [max_re_amr_len x batch x src_len]
        def handle_enc(enc):
            # src_enc : DoublePackedSequence:  [packed(pack_gold_amr_len x src_len x (src_enc_dim +1)), amr_lens, packed.data], 1 is indicator
            # unpacked_enc : list(batch_size, (real_g_amr_len x src_len x (src_enc_dim +1))
            # length_pairs : list(batch_size, (real_g_amr_len x src_len))
            unpacked_enc,length_pairs = doubleunpack(enc)
            # dim : src_enc_size + 1 , 1 is indicator
            dim = unpacked_enc[0].size(-1)
            weighted = []
            new_length_pairs = []
            for i, src_enc_t in enumerate(unpacked_enc):
                # p is a matrix with only real data in posterior
                p = posterior[:lengths[i],i,:] #re_real_amr_real_len  x src_len
                # src_enc_t (real_g_amr_len x src_len x (src_enc_dim+1))
                # enc_trans (src_len, real_g_amr_len x (src_enc_dim+1)), dim = src_enc_dim + 1
                enc_trans = src_enc_t.transpose(0,1).contiguous().view(p.size(-1),-1) #src_len x (pre_amr_len  x dim)
                weighted_enc = p.mm(enc_trans)   # re_real_amr_leon x (real_g_amr_len  x dim)
                weighted.append(weighted_enc.view(lengths[i],length_pairs[i][0],dim).transpose(0,1).contiguous())  #real_g_amr_len x real_re_amr_len  x dim
                # new_lengths = (real_g_amr_len, real_re_amr_len)
                new_length_pairs.append([length_pairs[i][0],lengths[i]])
            # doublepack(list(batch, g_amr_len x re_amr_len x dim), list(g_amr_l, re_amr_l))
            # DoublePackedSequence(packed: packed_g_amr_len x re_amr_len x dim), length= re_lens)
            return doublepack(weighted,length_pairs)

        return handle_enc(src_enc)

