#!/usr/bin/env python3.6
# coding=utf-8
'''

Iterating over data set, with bert support

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-04-30
'''
from utility.constants import *
from utility.data_helper import *
import torch
import math
from torch.nn.utils.rnn import PackedSequence
from parser.modules.helper_module import MyPackedSequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
from parser.DataIterator import *
import re
import logging
logger = logging.getLogger("mrp.BertDataIterator")

end= re.compile(".txt\_[a-z]*")

class BertDataIterator(DataIterator):

    def __init__(self, filePathes,opt,rel_dict,all_data = None):
        self.srcBert = []
        self.srcBertIndex = []
        super(BertDataIterator, self).__init__(filePathes, opt, rel_dict, all_data)

    def addSource(self, data):
        """
        add src ids, and original source
        for original source, it add the anchors of each token also
        tok, lem, pos, ner, anchors
        """
        # read input
        self.example_ids.append(data["example_id"])
        #src: snt_length x n_feature, contiguous means in memory in C order
        self.src.append(torch.LongTensor([data["snt_id"],data["lemma_id"],data["pos_id"],data["ner_id"]]).t().contiguous())
        self.src_char.append(torch.LongTensor(data['char_id']).contiguous())
        self.srcBert.append(torch.LongTensor(data['bert_id']).contiguous())
        self.srcBertIndex.append(torch.LongTensor(data['tok2bert_index']).contiguous())
        #source, before preprocessing into tensor, includes labels and tokens
        self.src_source.append([data["tok"],data["lem"],data["pos"],data["ner"],data["mwe"],data["anchors"]])

    #out : batch_size x max_bert_len
    def _batchify_srcBert(self, data, max_bert_seq_length):
        """
        Attention, this function will also clamp the last dimention offset in srcBatch towards the max_bert_length
        """
        #  batch_size, max_bert_length,
        out = data[0].new(len(data), max_bert_seq_length).fill_(BERT_PAD_INDEX)
        # narrow(dimension, start, length), for every sentence
        for i in range(len(data)):
            data_t = data[i].clone()
            assert data[i].size(0) <=  max_bert_seq_length, "check the bert tokenization, length exceed to max_bert_seq_length = {}".format(max_bert_seq_length)
            expanded_data_t = data_t.new(max_bert_seq_length).fill_(BERT_PAD_INDEX)
            expanded_data_t.narrow(0, 0, data_t.size(0)).copy_(data_t)
            out[i].narrow(0, 0, max_bert_seq_length).copy_(expanded_data_t)
        return out

    #out : batch_size x token_len x subword_len
    def _batchify_srcBertIndex(self, data, max_token_length):
        """
        Attention, this function will also clamp the last dimention offset in srcBatch towards the max_bert_length
        """
        #  batch_size, token_len x subword_len,
        max_sub_word_len = max([d.size(1) for d in data])
        out = data[0].new(len(data), max_token_length, max_sub_word_len).fill_(BERT_PAD_INDEX)
        # narrow(dimension, start, length), for every sentence
        for i in range(len(data)):
            data_t = data[i].clone()
            data_length = data[i].size(0)
            expanded_data_t = data_t.new(data_length, max_sub_word_len).fill_(BERT_PAD_INDEX)
            expanded_data_t.narrow(1, 0, data_t.size(1)).copy_(data_t)
            out[i].narrow(0, 0, data_length).copy_(expanded_data_t)
        return out

    def __getitem__(self, index):
        """
        retrieve every batch from the preprocessed dataset
        """
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        lengths,max_len = self.getLengths(index )
        def tuple2tensor(t):
            # generate a not batch first
            b = torch.stack(t, 0).contiguous()
            if self.cuda:
                b = b.cuda()
            return b

        def wrap(b,l ):
            #batch, len, feature
            if b is None:
                return b
            b = torch.stack(b, 0).transpose(0,1).contiguous()
            if self.cuda:
                b = b.cuda()
            # src_len, batch, feature
            packed =  pack(b,list(l))
            return packed

        def wrap_align(b,l ):
            #batch, len_tgt, len_src
            if b is None:
                return b
            b = torch.stack(b, 0).transpose(0,1).contiguous().float()
            if self.cuda:
                b = b.cuda()
            packed =  pack(b,list(l))
            #len_tag, batch, len_src
            return packed

        idsBatch = self.example_ids[index*self.batchSize:(index+1)*self.batchSize]
        # prep a tensor with fixed max_len of this batch
        # batch_size, max_len, n_feature
        srcBatch = self._batchify_src(
            self.src[index*self.batchSize:(index+1)*self.batchSize],max_len)

        # batch_size x word_len x char_len
        src_charBatch= self._batchify_src_char(
            self.src_char[index*self.batchSize: (index+1)*self.batchSize], max_len)

        # batch_size, max_bert_len, no need to get length for bert, all non-zero token are padding
        srcBertBatch = self._batchify_srcBert(
            self.srcBert[index*self.batchSize: (index+1)*self.batchSize], self.opt.max_bert_seq_length)

        srcBertIndexBatch = self._batchify_srcBertIndex(
            self.srcBertIndex[index*self.batchSize: (index+1)*self.batchSize], max_len)

        if self.source_only:
            src_sourceBatch = self.src_source[index*self.batchSize:(index+1)*self.batchSize]

            # zip with batch id
            batch = zip(idsBatch, srcBatch, src_charBatch, src_sourceBatch, srcBertBatch, srcBertIndexBatch)
            # sort by length, ascending
            order_data = sorted(list(enumerate(list(zip(batch, lengths)))),key = lambda x:-x[1][1])
            # order is the sorted index for batch
            order,data = zip(*order_data)
            batch, lengths = zip(*data)
            #keep consistent
            idsBatch, srcBatch, src_charBatch ,src_sourceBatch, srcBertBatch, srcBertBatchIndex = zip(*batch)
            return order, idsBatch, wrap(srcBatch,lengths),wrap(src_charBatch, lengths), src_sourceBatch, tuple2tensor(srcBertBatch), wrap(srcBertBatchIndex, lengths)

        else:
            # batch input data for amr
            tgtBatch = self._batchify_tgt(
                    self.tgt[index*self.batchSize:(index+1)*self.batchSize],max_len)
            # batch input for alignment from align_index
            alignBatch = self._batchify_align(
                    self.align_index[index*self.batchSize:(index+1)*self.batchSize],max_len)

            rel_seq_pre = self.rel_seq[index*self.batchSize:(index+1)*self.batchSize]
            rel_index_pre = self.rel_index[index*self.batchSize:(index+1)*self.batchSize]
            rel_role_pre = self.rel_mat[index*self.batchSize:(index+1)*self.batchSize]

            roots = self.root[index*self.batchSize:(index+1)*self.batchSize]

            src_sourceBatch = self.src_source[index*self.batchSize:(index+1)*self.batchSize]
            tgt_sourceBatch = self.tgt_source[index*self.batchSize:(index+1)*self.batchSize]
            sourceBatch = [  src_s +tgt_s for src_s,tgt_s in zip(src_sourceBatch,tgt_sourceBatch)]
            # within batch sorting by decreasing length for variable length rnns
            indices = range(len(srcBatch))

            # align with each data in a batch
            batch = zip(indices, idsBatch, srcBatch ,src_charBatch, tgtBatch,alignBatch,rel_seq_pre,rel_index_pre,rel_role_pre,sourceBatch,roots, srcBertBatch, srcBertIndexBatch)
            order_data =    sorted(list(enumerate(list(zip(batch, lengths)))),key = lambda x:-x[1][1])
            order,data = zip(*order_data)
            batch, lengths = zip(*data)
            indices, idsBatch, srcBatch, src_charBatch,tgtBatch,alignBatch ,rel_seq_pre,rel_index_pre,rel_role_pre,sourceBatch,roots,srcBertBatch, srcBertIndexBatch = zip(*batch)

            rel_batch,rel_index_batch,rel_lengths = self._batchify_rel_concept(rel_seq_pre,rel_index_pre)
            rel_roles,length_squares = self._batchify_rel_roles(rel_role_pre)

    #,wrap(charBatch))
            return order, idsBatch, wrap(srcBatch,lengths), wrap(src_charBatch, lengths), wrap(tgtBatch,lengths), wrap_align(alignBatch,lengths),\
                   MyPackedSequence(rel_batch,rel_lengths),rel_index_batch,MyPackedSequence(rel_roles,length_squares),roots,sourceBatch,tuple2tensor(srcBertBatch),wrap(srcBertIndexBatch, lengths)


    def shuffle(self):
    #    if True: return
        if self.source_only: #if data set if for testing
            data = list(zip(self.example_ids, self.src, self.src_char, self.src_source,self.srcBert, self.srcBertIndex))
            self.example_ids, self.src, self.src_char, self.src_source,self.srcBert, self.srcBertIndex= zip(*[data[i] for i in torch.randperm(len(data))])
        else:
            data = list(zip(self.example_ids, self.src, self.src_char, self.tgt,self.align_index,self.rel_seq,self.rel_index,self.rel_mat,self.root,self.src_source,self.tgt_source, self.srcBert, self.srcBertIndex))
            self.example_ids, self.src, self.src_char, self.tgt,self.align_index,self.rel_seq,self.rel_index,self.rel_mat,self.root,self.src_source,self.tgt_source, self.srcBert, self.srcBertIndex = zip(*[data[i] for i in torch.randperm(len(data))])
