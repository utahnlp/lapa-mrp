#!/usr/bin/env python3.6
# coding=utf-8
'''

Iterating over data set

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30

@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-06-30
'''
from utility.constants import *
from utility.data_helper import *
import torch
import math
from torch.nn.utils.rnn import PackedSequence
from parser.modules.helper_module import MyPackedSequence
from torch.nn.utils.rnn import pack_padded_sequence as pack
import re
end= re.compile(".txt\_[a-z]*")
import logging
logger = logging.getLogger("mrp.DataIterator")

def rel_to_batch(rel_batch_p,rel_index_batch_p,data_iterator,dicts,frame):
    """
    build rel batch with concept batches.
    rel_batch_p: concept batches
    rel_index_batch_p : alignment batches.
    """
    if frame =="amr":
        lemma_dict,amr_category_dict = dicts["lemma_dict"], dicts["amr_category_dict"]
        data = [torch.LongTensor([[amr_category_dict[uni.cat],lemma_dict[uni.le],0] for uni in uni_seq]) for uni_seq in rel_batch_p ]
    elif frame=="dm":
        target_pos_dict,cat_dict = dicts["dm_target_pos_dict"], dicts["dm_cat_dict"]
        data = [torch.LongTensor([[target_pos_dict[uni.pos],cat_dict[uni.cat],0] for uni in uni_seq]) for uni_seq in rel_batch_p ]
    elif frame =="psd":
        psd_target_pos_dict,psd_sense_dict = dicts["psd_target_pos_dict"], dicts["psd_sense_dict"]
        data = [torch.LongTensor([[psd_target_pos_dict[uni.pos],psd_sense_dict[uni.sense],0] for uni in uni_seq]) for uni_seq in rel_batch_p ]
    else:
        raise NotImplementedError("{} is not supported".format(frame))

    rel_index = [torch.LongTensor(index) for index in rel_index_batch_p]
    rel_batch,rel_index_batch,rel_lengths = data_iterator._batchify_rel_concept(data,rel_index)
    return  MyPackedSequence(rel_batch,rel_lengths),rel_index_batch

def role_mat_to_sparse(role_mat,rel_dict):
    """
    read a training data and make it tensors
    return a matrix, real_gold_amr_len x real_gold_amr_len
    """
    index =[]
    value = []
    # i is the index of head node, it is index from gold amr node
    for i,role_list in enumerate(role_mat):
        # role_list is list of [role_str, node2GoldIndex]
        for role_index in role_list:
            # if roleStr in rel_dict, if not in rel_dict, it will not be used.
            # for training, it is impossible, we have add all of them in.
            # if  something duplicate here for the role_index[1], then duplicate will existed.
            if role_index[0] in rel_dict:
                index.append([i,role_index[1]])
                value.append(rel_dict[role_index[0]])
    # size is length x length
    size = torch.Size([len(role_mat),len(role_mat)])
    # roleStr value tensor [length x 1]
    v = torch.LongTensor(value)
    if len(v) == 0:
        # transpose.
        i = torch.LongTensor([[0,0]]).t()
        v = torch.LongTensor([0])
        return torch.sparse.LongTensor(i,v,size)

    i = torch.LongTensor(index).t()
    # i is index, v is value, size
    return torch.sparse.LongTensor(i,v,size)

class DataIterator(object):

    def __init__(self, filePathes,opt,rel_dict,all_data = None):
        self.cuda = opt.gpus[0] != -1
        self.rel_dict = rel_dict
        self.all = []
        self.opt = opt
       #     break

 #       self.all = sorted(self.all, key=lambda x: x[0])
        self.example_ids = []
        self.src = []
        self.tgt = []
        self.src_char = []
        self.align_index = []
        self.rel_seq = []
        self.rel_index = []
        self.rel_mat = []
        self.root = []
        self.src_source = []
        self.tgt_source = []
        self.rel_tgt = []
        if all_data:
            skipped = 0
            for data in all_data:
                if self.read_sentence(data):
                    pass
                else:
                    skipped = skipped + 1
                    logger.warn("{} has non relations existed".format(data["example_id"]))
            self.batchSize = len(self.src)
            self.numBatches = 1
        else:

            for filepath in filePathes:
                n = self.readFile(filepath)
            self.batchSize = opt.batch_size
            self.numBatches = math.ceil(len(self.src)/self.batchSize)

        self.source_only = len(self.tgt_source) == 0

    def read_sentence(self,data):
        """
        read from a data exmaple, and add its source and target
        """
        self.addSource(data)
        if self.checkLegal(data):
            self.addTarget(data)
            return True
        else:
            return False

    def checkLegal(self, data):
        if "amr_id" in data:
            return len(data["amr_rel_id"]) != 0 and len(data["amr_rel_index"]) != 0
        elif "dm_id" in data:
            return len(data["dm_rel_id"]) != 0 and len(data["dm_rel_index"]) != 0
        elif "psd_id" in data:
            return len(data["psd_rel_id"]) != 0 and len(data["psd_rel_index"]) != 0

    def addTarget(self, data):
        """
        adding target according to different mr
        "dm_rel_seq","dm_rel_triples","dm_convertedl_seq","dm_seq"
        """
        #tgt: length x n_feature, n_feture is 5, AMR_CAT, AMR_LE, AMR_AUX, AMR_SENSE, AMR_CAN_COPY
        # after make all node aligned to a word or NULL word, length is equal to the length of tokes.
        if "amr_id" in data:
            self.tgt.append(torch.LongTensor(data["amr_id"]))  # lemma,cat, lemma_sense,ner,is_high
            # align_index, simple append all the aligned index
            # align_index = [[[i1,ij],[i2,ik] ]]
            self.align_index.append(data["amr_index"]) # this index is also recategorized id

            amrl = len(data["amr_id"])
            for i in data["amr_rel_index"]:
                assert i <amrl,data
            #rel
            self.rel_seq.append(torch.LongTensor(data["amr_rel_id"]))  # lemma,cat, lemma_sense, the order is in gold amr node order
            self.rel_index.append(torch.LongTensor(data["amr_rel_index"])) # index of head node from  recategorized node order
            # here use rel dict to exchange the roleStr into id., mats is a matrix [real_gold_amr_len x real_gold_amr_len]
            mats = role_mat_to_sparse(data["amr_roles_mat"], self.rel_dict)

            self.rel_mat.append(mats)  #role, index
            self.root.append(data["amr_root"])  #role, index for gold amr nodes

            #source means raw contents before becoming a tensor
            self.tgt_source.append([data["amr_rel_seq"],data["amr_rel_triples"],data["amr_convertedl_seq"],data["amr_seq"]])
        elif "psd_id" in data:
            self.tgt.append(torch.LongTensor(data["psd_id"]))  # lemma,cat, lemma_sense,ner,is_high
            # align_index, simple append all the aligned index
            # align_index = [[[i1,ij],[i2,ik] ]]
            self.align_index.append(data["psd_index"])

            amrl = len(data["psd_id"])
            for i in data["psd_rel_index"]:
                assert i <amrl,data
            #rel
            self.rel_seq.append(torch.LongTensor(data["psd_rel_id"]))  # lemma,cat, lemma_sense
            self.rel_index.append(torch.LongTensor(data["psd_rel_index"]))
            # here use rel dict to exchange the roleStr into id.
            mats = role_mat_to_sparse(data["psd_roles_mat"], self.rel_dict)
            self.rel_mat.append(mats)  #role, index
            self.root.append(data["psd_root"])  #role, index

            #source means raw contents before becoming a tensor
            self.tgt_source.append([data["psd_rel_seq"],data["psd_rel_triples"],data["psd_convertedl_seq"],data["psd_seq"]])
        elif "dm_id" in data:
            self.tgt.append(torch.LongTensor(data["dm_id"]))  # lemma,cat, lemma_sense,ner,is_high
            # align_index, simple append all the aligned index
            # align_index = [[[i1,ij],[i2,ik] ]]
            self.align_index.append(data["dm_index"])

            amrl = len(data["dm_id"])
            for i in data["dm_rel_index"]:
                assert i <amrl,data
            #rel
            self.rel_seq.append(torch.LongTensor(data["dm_rel_id"]))  # lemma,cat, lemma_sense
            self.rel_index.append(torch.LongTensor(data["dm_rel_index"]))
            # here use rel dict to exchange the roleStr into id.
            mats = role_mat_to_sparse(data["dm_roles_mat"], self.rel_dict)
            self.rel_mat.append(mats)  #role, index
            self.root.append(data["dm_root"])  #role, index

            #source means raw contents before becoming a tensor
            self.tgt_source.append([data["dm_rel_seq"],data["dm_rel_triples"],data["dm_convertedl_seq"],data["dm_seq"]])


    def addSource(self, data):
        """
        add src ids, and original source
        for original source, it add the anchors of each token also
        tok, lem, pos, ner, anchors
        """
        # read input
        self.example_ids.append(data["example_id"])
        self.src_char.append(torch.LongTensor(data['char_id']).contiguous())
        #src: snt_length x n_feature, contiguous means in memory in C order
        self.src.append(torch.LongTensor([data["snt_id"],data["lemma_id"],data["pos_id"],data["ner_id"]]).t().contiguous())
        #source, before preprocessing into tensor, includes labels and tokens
        if "mwe" not in data:
            data["mwe"] = 'O' * len(data["tok"])
        self.src_source.append([data["tok"],data["lem"],data["pos"],data["ner"],data["mwe"],data["anchors"]])


    def readFile(self,filepath):
        """
        For training and dev set, load all data, and make them tensors
        """
        logger.info("reading "+filepath)
        data_file = Pickle_Helper(filepath)

        all_data = data_file.load()["data"]
        skipped = 0
        for data in all_data:
            if self.read_sentence(data):
                pass
            else:
                skipped = skipped + 1
                logger.warn("{} has non relations existed".format(data["example_id"]))

        logger.info(("done reading {}, {} sentences processed, {} is skipped because of no relation").format(filepath, str(len(all_data)), skipped))
        return len(all_data)

    def _batchify_align(self, align_index,max_len):
        """
        align_index: batch_size x var(tgt_len) x [], every node, may have multiple index aligned.
        out : batch_size x tgt_len x src_len, alignment tensor, for every amr node, it shows wether every words can be aligned to every node.
        """
        out = torch.ByteTensor(len(align_index),max_len,max_len).fill_(0)
        for i in range(len(align_index)):
            # i for the ith sentence
            for j in range(len(align_index[i])):
                # j for the length of ith sentence
                if align_index[i][j][0] == -1:
                    out[i][j][:align_index[i][j][1]].fill_(1)
                else:
                    # for every index k, which is possible word can be aligned, make it boolean 1
                    for k in align_index[i][j]:
                        out[i][j][k] = 1
            # for padding node, fill all ones
            for j in range(len(align_index[i]),max_len):   #for padding
                out[i][j][len(align_index[i]):].fill_(1)
        return out

    #rel_seq: batch_size x var(len) x n_feature
    #rel_index: batch_size x var(len)

    #out : all_data x n_feature
    #out_index: batch_size x var(len)
    #lengths : batch_size
    def _batchify_rel_concept(self, data,rel_index ):
        lengths = [len(x) for x in data]
        for l in lengths:
            assert l > 0, (data,rel_index)
        second = max([x.size(1) for x in data])
        total = sum(lengths)
        out = data[0].new(total, second)
        out_index = []
        current = 0
        for i in range(len(data)):
            data_t = data[i].clone()
            out.narrow(0, current, lengths[i]).copy_(data_t)
            index_t = rel_index[i].clone()
            if self.cuda:
                index_t = index_t.cuda()
            out_index.append(index_t)
          #  out_index.append(index_t)
            current += lengths[i]
        if self.cuda:
            out = out.cuda()
        return out,out_index,lengths


    #rel_mat: batch_size x var(len) x var(len)
    #rel_index: batch_size x var(len)

    #out :  (batch_size x var(len) x var(len))
    def _batchify_rel_roles(self, all_data ):
        length_squares = [x.size(0)**2 for x in all_data]
        total = sum(length_squares)
        out = torch.LongTensor(total)
        current = 0
        for i in range(len(all_data)):
            # before to dense
            # https://pytorch.org/docs/stable/sparse.html
            # after to dense, if duplicate indicse, then the value will be added.
            data_t = all_data[i].to_dense().clone().view(-1)
            out.narrow(0, current, length_squares[i]).copy_(data_t)
            current += length_squares[i]

        if self.cuda:
            out = out.cuda()

        return out,length_squares


    #data: batch_size x var(len) x n_feature
    #out : batch_size x tgt_len x n_feature
    def _batchify_tgt(self, data,max_src ):
        lengths = [x.size(0) for x in data]
        max_length = max(max(x.size(0) for x in data),max_src)   #if y, we need max_x
        out = data[0].new(len(data), max_length,data[0].size(1)).fill_(PAD)
        for i in range(len(data)):
            data_t = data[i].clone()
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data_t)
        return out

    #data: batch_size x var(len) x n_feature
    #out : batch_size x src_len x n_feature
    def _batchify_src(self, data,max_length ):
        #  batch_size, max_length, ,
        out = data[0].new(len(data), max_length,data[0].size(1)).fill_(PAD)
        # narrow(dimension, start, length), for every sentence
        for i in range(len(data)):
            data_t = data[i].clone()
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data_t)
        return out

    #out : batch_size x token_len x char_len
    def _batchify_src_char(self, data, max_token_length):
        #  batch_size, token_len x char_len,
        max_char_len = max([d.size(1) for d in data])
        out = data[0].new(len(data), max_token_length, max_char_len).fill_(PAD)
        # narrow(dimension, start, length), for every sentence
        for i in range(len(data)):
            data_t = data[i].clone()
            data_length = data[i].size(0)
            expanded_data_t = data_t.new(data_length, max_char_len).fill_(PAD)
            expanded_data_t.narrow(1, 0, data_t.size(1)).copy_(data_t)
            out[i].narrow(0, 0, data_length).copy_(expanded_data_t)
        return out

    def getLengths(self,index):
        src_data = self.src[index*self.batchSize:(index+1)*self.batchSize]
        src_lengths = [x.size(0) for x in src_data]
        if  self.source_only:
            return src_lengths,max(src_lengths)

        tgt_data = self.tgt[index*self.batchSize:(index+1)*self.batchSize]
        tgt_lengths = [x.size(0) for x in tgt_data]
        lengths = []
        for i,l in enumerate(src_lengths):
            lengths.append(max(l,tgt_lengths[i]))
        return lengths,max(lengths)

    def __getitem__(self, index):
        """
        retrieve every batch from the preprocessed dataset
        """
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        lengths,max_len = self.getLengths(index )
        def wrap(b,l ):
            #batch, len, feature
            if b is None:
                return b
            b = torch.stack(b, 0).transpose(0,1).contiguous()
            if self.cuda:
                b = b.cuda()
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
            return packed

        idsBatch = self.example_ids[index*self.batchSize:(index+1)*self.batchSize]
        # prep a tensor with fixed max_len of this batch
        srcBatch = self._batchify_src(
            self.src[index*self.batchSize:(index+1)*self.batchSize],max_len)

        # batch_size x word_len x char_len
        src_charBatch= self._batchify_src_char(
            self.src_char[index*self.batchSize: (index+1)*self.batchSize], max_len)

        if self.source_only:
            src_sourceBatch = self.src_source[index*self.batchSize:(index+1)*self.batchSize]

            batch = zip(idsBatch, srcBatch, src_charBatch, src_sourceBatch)
            order_data =    sorted(list(enumerate(list(zip(batch, lengths)))),key = lambda x:-x[1][1])
            order,data = zip(*order_data)
            batch, lengths = zip(*data)
            idsBatch, srcBatch, src_charBatch, src_sourceBatch = zip(*batch)
            return order,idsBatch, wrap(srcBatch,lengths), wrap(src_charBatch, lengths), src_sourceBatch

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

            roots =self.root[index*self.batchSize:(index+1)*self.batchSize]

            src_sourceBatch = self.src_source[index*self.batchSize:(index+1)*self.batchSize]
            tgt_sourceBatch = self.tgt_source[index*self.batchSize:(index+1)*self.batchSize]
            sourceBatch = [  src_s +tgt_s for src_s,tgt_s in zip(src_sourceBatch,tgt_sourceBatch)]
            # within batch sorting by decreasing length for variable length rnns
            indices = range(len(srcBatch))

            batch = zip(indices, idsBatch, srcBatch, src_charBatch, tgtBatch,alignBatch,rel_seq_pre,rel_index_pre,rel_role_pre,sourceBatch,roots)
            order_data =    sorted(list(enumerate(list(zip(batch, lengths)))),key = lambda x:-x[1][1])
            order,data = zip(*order_data)
            batch, lengths = zip(*data)
            indices, idsBatch, srcBatch,src_charBatch, tgtBatch,alignBatch ,rel_seq_pre,rel_index_pre,rel_role_pre,sourceBatch,roots= zip(*batch)

            rel_batch,rel_index_batch,rel_lengths = self._batchify_rel_concept(rel_seq_pre,rel_index_pre)
            rel_roles,length_squares = self._batchify_rel_roles(rel_role_pre)


            return order,idsBatch, wrap(srcBatch,lengths), wrap(src_charBatch, lengths), wrap(tgtBatch,lengths), wrap_align(alignBatch,lengths),\
                   MyPackedSequence(rel_batch,rel_lengths),rel_index_batch,MyPackedSequence(rel_roles,length_squares),roots,sourceBatch

    def __len__(self):
        return self.numBatches


    def shuffle(self):
    #    if True: return
        if self.source_only: #if data set if for testing
            data = list(zip(self.example_ids, self.src, self.src_char, self.src_source))
            self.example_ids, self.src, self.src_char, self.src_source = zip(*[data[i] for i in torch.randperm(len(data))])
        else:
            data = list(zip(self.example_ids, self.src, self.src_char, self.tgt,self.align_index,self.rel_seq,self.rel_index,self.rel_mat,self.root,self.src_source,self.tgt_source))
            self.example_ids, self.src, self.src_char,self.tgt,self.align_index,self.rel_seq,self.rel_index,self.rel_mat,self.root,self.src_source,self.tgt_source = zip(*[data[i] for i in torch.randperm(len(data))])

