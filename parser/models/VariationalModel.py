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
from parser.models.DMGraphModel import *
from parser.models.PSDGraphModel import *
from parser.modules.encoder_zoo import *

from parser.modules.GumbelSoftMax import renormalize,sink_horn,gumbel_noise_sample
from parser.modules.helper_module import *

from pytorch_transformers.modeling_bert import BertModel
from allennlp.modules.seq2seq_encoders.stacked_self_attention import *
import json

from copy import deepcopy
import logging
logger = logging.getLogger("mrp")
#Model to compute relaxed posteior
# we constraint alignment if copying mechanism can be used
class Posterior(nn.Module):
    def __init__(self,opt, src_enc_size, amr_enc_size):
        super(Posterior, self).__init__()
        # when using GCN, change the amr_rnn_size to some amr_encode size
        self.opt = opt
        self.txt_enc_size = src_enc_size
        self.amr_enc_size = amr_enc_size
        self.scale = 1.0 / np.sqrt(self.amr_enc_size)
        self.jamr = opt.jamr
        self.gumbel = opt.gumbel
        if self.jamr : #if use fixed alignment, then no need for variational model
            return
        self.transform = nn.Sequential(
            nn.Dropout(opt.dropout),
            nn.Linear(self.txt_enc_size,self.amr_enc_size,bias = opt.lemma_bias))
        self.sm = nn.Softmax(dim=-1)
        self.sink = opt.sink
        self.sink_t = opt.sink_t
        self.mask_pre_unaligned = opt.mask_pre_unaligned
        if opt.cuda:
            self.cuda()

    def forward(self,src_enc,amr_enc,aligns):

        '''src_enc: ((src_len x  batch), txt_rnn_size), src_l
           amr_enc: ((amr_len x  batch), amr_rnn_size), amr_l
            aligns: ((amr_len x  batch), src_len) , amr_l


            posterior: amr_len x  batch x src_len , amr_l
        '''
        if self.jamr :
            return aligns,aligns,0
        # unpack make it pad_packed_sequence
        src_enc_tuple,amr_enc_tuple,aligns_tuple =unpack(src_enc),unpack(amr_enc),unpack(aligns)

        # max_src_len x batch x txt_rnn_size
        src_enc_tensor = src_enc_tuple[0]
        # max_amr_len x batch x amr_enc_size
        amr_enc_tensor = amr_enc_tuple[0]
        # different lenght of amr_len
        lengths = aligns_tuple[1]
        # max_amr_len x batch x src_len
        aligns_tensor = aligns_tuple[0]
        assert not np.isnan(np.sum(src_enc_tensor.detach().cpu().numpy())),("src_enc \n",src_enc_tensor.detach())
        assert not np.isnan(np.sum(amr_enc_tensor.detach().cpu().numpy())),("amr_enc \n",amr_enc_tensor.detach())
        src_len , batch , src_enc_size = src_enc_tensor.size()
        src_transformed = self.transform(src_enc_tensor.view(-1,src_enc_size)).view(src_len, batch, -1).transpose(0,1)        #batch  x  src_len x  amr_enc_size
        assert not np.isnan(np.sum(src_transformed.detach().cpu().numpy())),("amr_enc \n",src_transformed.detach())
        amr_transformed = amr_enc_tensor.transpose(0,1).transpose(1,2) #batch    x amr_encode_size x  amr_len
        assert not np.isnan(np.sum(amr_transformed.detach().cpu().numpy())),("amr_enc \n",amr_transformed.detach())
        #logger.info("bmWeights:{}\n".format(
        #    list(self.transform.modules())[2].weight.detach().cpu().numpy()
        #))
        # batch matrix matric product, score: amr_len x batch x src_len
        score = src_transformed.bmm(amr_transformed).transpose(1,2).transpose(0,1) #/ self.amr_rnn_size  #amr_len x batch  x  src_len
        score = score * self.scale
        assert not np.isnan(np.sum(score.detach().cpu().numpy())),("score \n",score)
        if self.gumbel:
            final_score = gumbel_noise_sample(score)  if self.training else score
        else:
            final_score = score

        assert not np.isnan(np.sum(final_score.detach().cpu().numpy())),("final_score \n",final_score)

        if self.sink:
            # also adding prealigns as priors, make those manual aligned part in high score, other score are penaltied
            posterior = sink_horn((final_score- (1-aligns_tensor)*self.mask_pre_unaligned,lengths),k=self.sink,t=self.sink_t )
        else:
            final_score = final_score- (1-aligns_tensor)*self.mask_pre_unaligned
            # dim : (amr_len, batch, src_len)
            dim = final_score.size()
            final_score = final_score.view(-1, final_score.size(-1))
            posterior =self.sm(final_score).view(dim)
        return pack(posterior, lengths),pack(score,lengths) #amr_len x batch  x  src_len

#directly compute likelihood of concept being generated at words (a matrix for each training example)
def LikeliHood(tgtBatch,probBatch,frame):
    '''tgtBatch:  data x  [n_feature + 1 (AMR_CAN_COPY)], batch_sizes
        probBatch: (data x n_out, lengths ) * , cat, le, ner
            aligns:  (amr_len x  batch), src_len , amr_l

        likelihood: data (amr) x src_len , batch_sizes
    '''
    # different amr lens
    batch_sizes = tgtBatch.batch_sizes
    #tgtBatch is not unpacked, tgtBatch.data : [data x n_features]
    likelihoods = []
    # probBatch: pack(src_len x batch, cat), pack(src_len x batch_size, le), pack(src_len x batch_size, ner)
    for i,prob in enumerate(probBatch):
        assert isinstance(prob, PackedSequence),"only support packed"
        #if (frame == "amr" and i == AMR_LE ) or (frame == "psd" and i==PSD_LE ) or ( frame == "dm" and i in [DM_CAT, DM_LE]):
        if (frame == "amr" and i == AMR_LE ) or (frame == "psd" and i==PSD_LE ) or ( frame == "dm" and i in [DM_CAT]):
            # AMR_LE == TXT_LEMMMA == 1
            # prob_batch: [max_src_len x batch_size x n_out], lengths: src_len
            prob_batch,lengths = unpack(prob) # prob_batch = [src_len x batch x n_out], lengths=[src_len]
            # prob_batch: [batch_size x max_src_len x n_out], lengths: src_len
            # max_src_len == max_amr_len when make batch in data iterater.
            prob_batch = prob_batch.transpose(0,1)  #  batch x src_len x n_out
            #logger.info("le_prob_batch:{}".format(prob_batch))
            n_out = prob_batch.size(-1)
            src_len = prob_batch.size(1) # the max src length
            # unknow lemma, pruned by frequency, and make the last one as it label
            # packed_index_data : [data(amr), 1]
            # Here likihood is not calculated from expanded node, but from the converted node
            # which means, lemma large than n_out -1, will hope the copy flag can help.
            # But the can copy is not a seperate prob that can be multiplied, instead, the probiity is added to the prob only when the tgtBatch can copy, other wise there is no impact
            packed_index_data = tgtBatch.data[:,i].clamp(max=n_out-1) #so lemma not in high maps to last index ,data x 1
            if frame == "amr":
                copy_data = (packed_index_data<n_out-1).float()*tgtBatch.data[:,AMR_CAN_COPY].float()
            elif frame == "psd":
                # make packed_index_data only consider the high_dict part. Because only them are classified, and learnable.
                # index beyond that can be copied by any rules
                # for any likelihood beyond the high_dict, we directly think that likihood is 1
                copy_data = (packed_index_data<n_out-1).float()*tgtBatch.data[:,PSD_CAN_COPY].float()
            elif frame == "dm":
                if i == DM_CAT:
                    copy_data = (packed_index_data<n_out-1).float()*tgtBatch.data[:,DM_CAN_COPY].float()
                #elif i == DM_LE:
                #    copy_data = (packed_index_data<n_out-1).float()*tgtBatch.data[:,DM_LE_CAN_COPY].float()

            likes = []
            used = 0 # for packed sequence
            for batch_size in batch_sizes:  #n_different length in amr

                # likeihood, [topK, src_len, n_out]
                likelihood = prob_batch[:batch_size]
                # batch_index : the corresponding lemma index, not the high index, still only consider the part in high _dict
                batch_index = packed_index_data[used:used+batch_size].view(batch_size,1,1).expand(batch_size,src_len,1) #batch_size x src_len x 1
                # like0 [batch_szie, src_len]
                # only select P(c|w)
                like0 = likelihood.gather(2,batch_index).squeeze(2) #+  likelihood[:,:,-1]*high_and_copy    # will be corrected by posterior/align
                pointer = copy_data[used:used+batch_size].view(batch_size,1).expand(batch_size,src_len) # *alignBatch.data[used:used+batch_size] #batch_size x src_len x 1
                # either from copying the candidates or over potential categorizs.
                copy_prob = likelihood[:,:,-1]
                like = copy_prob*pointer + like0 # batch_size x src_len
                likes.append(like) #batch_size x src_len
                used = used + batch_size
            packed_likes = torch.cat(likes,0) #data x src_len

            likelihoods.append(packed_likes)
        else:
            prob_batch,lengths = unpack(prob)
            prob_batch = prob_batch.transpose(0,1)  #  batch x src_len x n_out
            src_len = prob_batch.size(1)
            packed_index_data = tgtBatch.data[:,i].contiguous()
            likes = []
            used = 0
       #     print (i,packed_index_data,batch_sizes)
            for batch_size in batch_sizes:  #n_different length in amr
                likelihood = prob_batch[:batch_size]   #batch_size x src_len x n_out
                batch_index = packed_index_data[used:used+batch_size].view(batch_size,1,1).expand(batch_size,src_len,1) #batch_size x src_len x 1
                # For ner, cat, it is determistic, without copy mechanism.
                # logger.info("likelihood:{}, batch_index: {}".format(likelihood.size(), batch_index))
                likes.append(likelihood.gather(2,batch_index).squeeze(2)) #batch_size x src_len
                used = used + batch_size
            packed_likes = torch.cat(likes,0) #data x src_len
            likelihoods.append(packed_likes)
    likelihood = 1
    # likelihoods contains, cat, le, ner_prob
    for i in range(0,len(likelihoods)):
        likelihood = likelihood  * likelihoods[i]
    # ndata(amr) x src_len
    return PackedSequence(likelihood,batch_sizes)

#compute variational posterior, and corresponding likelihood for concept identification, those are needed for computing loss
#also return the gumbel-sinkhorn input score matrix, which is needed for regularization
class VariationalAlignmentModel(nn.Module):

    def __init__(self, opt,embs, encoder = None, component_dict = {}, frame="amr"):
        super(VariationalAlignmentModel, self).__init__()
        self.component_dict = component_dict
        self.frame = frame
        if encoder:
            self.snt_encoder = encoder
        else:
            self.snt_encoder = EncoderZoo.create_sentence_encoder_from_config(embs, self.component_dict, opt.posterior_snt_encoder, opt)
        # when with alignment, use opt.jamr
        self.jamr = opt.jamr
        if self.jamr :
            return

        self.init_target_encoder(opt, embs)
        self.posterior = Posterior(opt, src_enc_size=self.snt_encoder.src_enc_size, amr_enc_size=self.amr_encoder.amr_enc_size)

    def init_target_encoder(self, opt, embs):
        amr_encode_configs = opt.posterior_amr_encoder.split(":")
        amr_component_type = amr_encode_configs[0]
        if len(amr_encode_configs) > 1:
            amr_component_id = amr_encode_configs[1]
        if opt.posterior_amr_encoder.startswith("rnn"):
            if self.frame == "amr":
                self.amr_encoder = AmrEncoder( opt, embs)
            elif self.frame == "dm":
                self.amr_encoder = DMEncoder( opt, embs)
            elif self.frame =="psd":
                self.amr_encoder = PSDEncoder( opt, embs)
        elif opt.posterior_amr_encoder.startswith("transformer"):
            if amr_component_id in self.component_dict:
                transformer_encoder = self.component_dict[amr_component_id]
            else:
                remaining_amr_json_configs = ':'.join(amr_encode_configs[2:])
                transformer_encoder = create_transformer_encoder_from_json_config(remaining_amr_json_configs, opt)
                if opt.cuda > 0:
                    if opt.cuda == 2:
                        device_name = 'cuda:1'
                    else:
                        device_name = 'cuda:{}'.format((len(self.component_dict)+1)%opt.cuda)
                else:
                    device_name = 'cpu'
                transformer_encoder.to(torch.device(device_name))
                logger.info("{} initialized on device {}".format(opt.posterior_amr_encoder, device_name))
                self.component_dict[amr_component_id] = transformer_encoder

            if self.frame == "amr":
                self.amr_encoder = AmrTransformerEncoder( opt, embs, transformer_encoder)
            elif self.frame == "dm":
                self.amr_encoder = DMTransformerEncoder( opt, embs, transformer_encoder)
            elif self.frame =="psd":
                self.amr_encoder = PSDTransformerEncoder( opt, embs, transformer_encoder)
        else:
            raise NotImplementedError("opt.posterior_amr_encoder is not supported: {}".format(opt.posterior_amr_encoder))

    def forward(self, srcBatch, src_charBatch, tgtBatch,alignBatch,probBatch,bertBatch=None, bertIndexBatch=None):
        assert isinstance(srcBatch,PackedSequence)

        # likelihood for concept identification, will be used for prediction
        likeli = LikeliHood(tgtBatch,probBatch, self.frame)  # likelihood: data (amr) x src_len , batch_sizes
        if self.jamr :
            posterior,score = alignBatch,alignBatch
            posterior = PackedSequence(renormalize(posterior.data),posterior.batch_sizes)
            return posterior,likeli,score

        srcBatchData = srcBatch.data
        # if already encoded, when using rnn, it is txt_rnn_size
        # otherwise, it is bert_output_size, [batch_size, sequence_length, bert_hidden_size]
        if srcBatchData.size(-1) == self.snt_encoder.src_enc_size:
            src_enc = srcBatch
        else:
            # encode it now
            # src_enc : pack[(src_len x batch_size)x rnn_size, size]
            src_enc = self.snt_encoder(srcBatch, src_charBatch, bertBatch=bertBatch, bertIndexBatch=bertIndexBatch)

        assert not np.isnan(np.sum(src_enc.data.detach().cpu().numpy())),("src_enc\n",src_enc.data)
        # tgtBatch : packed(data(amr_len), n_features), lengs
        # amr_ec : packed(data(amr_len), amr_enc_size)
        amr_enc = self.amr_encoder(tgtBatch )
        assert not np.isnan(np.sum(amr_enc.data.detach().cpu().numpy())),("amr_enc\n",amr_enc.data)
        posterior,score = self.posterior(src_enc,amr_enc,alignBatch)
        # posterior: packed(data(amr_len)x src_len)
        return posterior,likeli,score
