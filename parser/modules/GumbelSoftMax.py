#!/usr/bin/env python3.6
# coding=utf-8
'''

Helper functions regarding gumbel noise

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence

eps = 1e-8
def sample_gumbel(input):
    with torch.no_grad():
        noise = torch.rand(input.size()).type_as(input)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        return noise


def gumbel_noise_sample(input,temperature = 1):
    noise = sample_gumbel(input)
    x = (input + noise) / temperature
    return x.view_as(input)


import numpy as np

np.seterr(all='raise')
def sink_horn(input,k = 5,t = 1,batch_first = False):
    def sink_horn_data(x,lengths):
        assert not np.isnan(np.sum(x.detach().cpu().numpy())),("start x\n",x)
        over_flow = x-80*t
        clamped_x = x.clamp(max=80*t)+torch.tanh(over_flow)*(over_flow>0).float()
        exp_clamped_x = torch.exp(clamped_x/t)
        #assert not np.isnan(np.sum(exp_clamped_x.detach().cpu().numpy())),("exp exp_clamped_x\n",exp_clamped_x)
        musks = torch.zeros(exp_clamped_x.size())
        # lengths is amr_len for each amr
        for i,l in enumerate(lengths):
            musks[:l,i,:l] = 1
        typed_musks = musks.type_as(exp_clamped_x)
        musked_x= exp_clamped_x*typed_musks+eps
        for i in range(0,k):
            # musked_x.sum(0, keepdim=True), [1, batch, src_len] -> [amr_len, batch, src_len]
            x1 = musked_x / musked_x.sum(0,keepdim=True).expand_as(musked_x)
            musked_x1 = x1*typed_musks+eps
            # musked_x1.sum(2, keepdim=True),[amr_len, batch, 1] -> [amr_len, batch, src_len]
            x2 = musked_x1 / musked_x1.sum(2,keepdim=True).expand_as(musked_x1)
            musked_x = x2*typed_musks+eps

        assert not np.isnan(np.sum(musked_x.detach().cpu().numpy())),("end musked_x\n",musked_x)
        return musked_x
    if isinstance(input,PackedSequence):
        # data is padded matrix
        data,l = unpack(input,batch_first=batch_first)
        output = sink_horn_data(data,l)
        return pack(output,l,batch_first)
    else:
        return sink_horn_data(*input)


def renormalize(input,t=1):

    x = ((input+eps).log() ) / t
    x = F.softmax(x, dim=-1)
    return x.view_as(input)

