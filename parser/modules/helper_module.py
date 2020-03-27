#!/usr/bin/env python3.6
# coding=utf-8
'''

Some data structure to save memory for packing variable lengthed data into batch,
Not actually sure whether it's better (time or space) than zero padding,

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''
import torch
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
import json
from collections import namedtuple
MyPackedSequence = namedtuple('MyPackedSequence', ['data', 'lengths'])
MyDoublePackedSequence = namedtuple('MyDoublePackedSequence', ['PackedSequence', 'length_pairs','data'])   #packed sequence must be batch_first, inner length
DoublePackedSequence = namedtuple('DoublePackedSequence', ['PackedSequence', 'outer_lengths','data'])   #packed sequence must be batch_first, inner length

# pytorch not support 0-length packed sequence
# https://github.com/pytorch/pytorch/issues/9681

def sort_index(seq):
    sorted([(v, i) for (i, v) in enumerate(seq)],reverse = True)

def mypack(data,lengths):
    if isinstance(data,list):
        return MyPackedSequence(torch.cat(data,0),lengths)
    else:
        data_list = []
        for i, l in enumerate(lengths):
            data_list.append(data[i][:l])
        return mypack(data_list,lengths)


def myunpack(*mypacked):
    data,lengths = mypacked
    data_list = []
    current = 0
    for i, l in enumerate(lengths):
        data_list.append(data[current:l+current])
        current += l
    return data_list

def mydoubleunpack(mydoublepacked):
    packeddata,length_pairs,data = mydoublepacked
    data = myunpack(*packeddata)
    data_list = []
    for i, ls in enumerate(length_pairs):
        out_l,in_l = ls
        data_list.append(data[i][:,:in_l])   #outl x max_l x dim
    return data_list,length_pairs


def mydoublepack(data_list,length_pairs):  #batch x var(amr_l x src_l x dim)
    data = []
    max_in_l = max([ls[1] for ls in length_pairs])
    outer_l = []
    for d, ls  in list(zip(data_list,length_pairs)):
        outl,inl = ls
        size = [i for i in d.size()]
        if size[1] == max_in_l:
            tdata = d
        else:
            size[1] = max_in_l
            tdata = d.new_zeros(size)
       #     print (tdata)
            # tdata = t_tdata.clone()
            tdata[:,:inl] = d

        data.append( tdata)   #amr_l x src_l x dim
        outer_l.append(outl)

    packed = mypack(data,outer_l)

    return MyDoublePackedSequence(packed,length_pairs,packed.data)

def doubleunpack(doublepacked):
    assert isinstance(doublepacked,DoublePackedSequence)
    packeddata,outer_lengths,data = doublepacked
    data,in_l = unpack(packeddata,batch_first=True)
    data_list = []
    length_pairs = []
    current = 0
    for i, l in enumerate(outer_lengths):
        data_list.append(data[current:l+current])   #outl x max_l x dim
        length_pairs.append((l,in_l[current]))
        current += l
    return data_list,length_pairs


def doublepack(data_list,length_pairs):  #batch x var(amr_l x src_l x dim)
    data = []
    lengths = []
    max_in_l = max([ls[1] for ls in length_pairs])
    outer_l = []
    for d, ls  in list(zip(data_list,length_pairs)):
        outl,inl = ls
        size = [i for i in d.size()]
        if size[1] == max_in_l:
            tdata = d
        else:
            size[1] = max_in_l
            tdata = d.new_zeros(size)
            tdata[:,:inl] = d
        data.append( tdata)   #amr_l x src_l x dim
        lengths = lengths + [inl]*outl
        outer_l.append(outl)

    packed = pack(torch.cat(data,0),lengths,batch_first=True)

    return DoublePackedSequence(packed,outer_l,packed.data)

def data_dropout(data,frequency,UNK = 1):
    if frequency == 0: return data
    if isinstance(frequency,torch.Tensor):
        f = frequency
        unk_mask = torch.bernoulli(f).to(data.device)
        data = data*(1-unk_mask).long()+(unk_mask*torch.ones(data.size(), device=data.device)*UNK).long()
    else:
        f = torch.ones(data.size())*frequency
        unk_mask = torch.bernoulli(f).to(data.device)
        data = data*(1-unk_mask).long()+(unk_mask*torch.ones(data.size(), device=data.device)*UNK).long()
    return data
