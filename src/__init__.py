

import torch
import torch.nn as nn
import sys
import os
import logging
import re
import parser
from src.train import * 

from copy import deepcopy


logger = logging.getLogger("mrp")

def freeze(m,t=0):
    if isinstance(m,nn.Dropout):
        m.p  = t
    # m.dropout =t

def load_old_model(frame, dicts,opt,generate=False):
    """
    TODO: to load from multiple models.
    """
    model_from = opt.restore_from
    logger.info('Loading from checkpoint at {}'.format(model_from))
    # gpus[0] != -1, then it means it use gpus, thenwe need to make it , -1 means cpu, -2, means it trust the CUDA_VISIBLE_DEVICE given by the system.
    if opt.gpus[0] == -1 : # cpu
        logger.info('from model in gpus:'+str(opt.from_gpus[0]),'to cpu ')
        # storage mapping from gpuid to cpu.
        checkpoint = torch.load(model_from, map_location={'cuda:'+str(opt.from_gpus[0]): 'cpu'})
    elif opt.gpus[0] == -2:
        # schedule by the system automatically, trust the CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"]:
            gpus_str = os.environ["CUDA_VISIBLE_DEVICES"]
            logger.info("trust the scheduler:" + os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            # default will use the first one, it is 0
            gpus_str = "0"
            logger.info("NO CUDA_VISIBLE_DEVICE found, use default gpu:0" + str(torch.cuda.device_count()))
        # Some GPU scheduler will broken by specifiying the CUDA_VISIBLE_DEVICES, if you need this, just uncomment it.
        # use 1,2,3 to set multiple gpuids
        opt.gpus=[int(i_str) for i_str in gpus_str.split(',') if i_str !='']
        restore_model_name=model_from.split('/')[-1]
        if opt.from_gpus:
            if not restore_model_name.startswith("gpus_{}".format(opt.from_gpus[0])):
                logger.info('opt.from_gpus[0] = {} is not consitent with restore model name {}'.format(opt.from_gpus[0], restore_model_name))
                logger.info('from model in gpus:'+str(opt.from_gpus[0]),' to gpu:'+str(opt.gpus[0]))
        else:
            opt.from_gpus = [ int(i) for i in re.findall(r'\d+', restore_model_name)]
            logger.info('opt.from_gpus is empty, set from restore_model_name:{}, from_gpus={}'.format(restore_model_name, opt.from_gpus))
        # storage mapping, between gpus
        checkpoint = torch.load(model_from, map_location={'cuda:'+str(opt.from_gpus[0]): 'cuda:'+str(opt.gpus[0])})
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(opt.gpus))
        logger.info("CUDA_VISIBLE_DEVICES speificed with {}".format(opt.gpus))
        # storage mapping, between gpus
        restore_model_name=model_from.split('/')[-1]
        if opt.from_gpus:
            if not restore_model_name.startswith("gpus_{}".format(opt.from_gpus[0])):
                logger.info('opt.from_gpus[0] = {} is not consitent with restore model name {}'.format(opt.from_gpus[0], restore_model_name))
                logger.info('from model in gpus:'+str(opt.from_gpus[0]),' to gpu:'+str(opt.gpus[0]))
        else:
            opt.from_gpus = [ int(i) for i in re.findall(r'\d+', restore_model_name)]
            logger.info('opt.from_gpus is empty, set from restore_model_name:{}, from_gpus={}'.format(restore_model_name, opt.from_gpus))
        # storage mapping, between gpus
        checkpoint = torch.load(model_from, map_location={'cuda:'+str(opt.from_gpus[0]): 'cuda:'+str(opt.gpus[0])})

    logger.info("Model loaded")
    optt = checkpoint["opt"]
    optim = checkpoint["optim"]
    rel = optt.rel

    embs = embedding_from_dicts( opt, dicts)
    MrpModel = parser.models.MRPModel(opt, embs, component_dict={}, frame = frame)
    MrpModel.load_state_dict(checkpoint['model'])
    if optt.rel == 1:
        if not opt.retrain_all:
            MrpModel.concept_decoder = deepcopy(MrpModel.concept_decoder)
            for name, param in MrpModel.concept_decoder.named_parameters():
                param.requires_grad = False
            MrpModel.concept_decoder.apply(freeze)

        parameters_to_train = []
        for name, param in MrpModel.named_parameters():
            if name == "word_fix_lut" or param.size(0) == len(dicts["word_dict"]):
                param.requires_grad = False
            if param.requires_grad:
                parameters_to_train.append(param)
        logger.info("MrpModel:\n{}".format(MrpModel))
        logger.info("training parameters: "+str(len(parameters_to_train)))
        return MrpModel,parameters_to_train,optt, optim

    optt.rel = opt.rel
    if opt.rel and not rel  :
        if opt.jamr == 0:
            MrpModel.poserior_m.align_weight = 1
        MrpModel.concept_decoder.apply(freeze)
        opt.independent = True
        MrpModel.start_rel(opt)
        embs = MrpModel.embs
        embs["lemma_lut"].requires_grad = False      ##need load
        embs["pos_lut"].requires_grad = False
        embs["ner_lut"].requires_grad = False
        embs["word_fix_lut"].requires_grad = False
        embs["rel_lut"] =   nn.Embedding(dicts["rel_dict"].size(),
                          opt.rel_dim)
        for param in MrpModel.concept_decoder.parameters():
            param.requires_grad = False
    if not generate and opt.jamr == 0:
        MrpModel.poserior_m.posterior.ST = opt.ST
        MrpModel.poserior_m.posterior.sink = opt.sink
        MrpModel.poserior_m.posterior.sink_t = opt.sink_t

    if opt.cuda:
        MrpModel.cuda()
    else:
        MrpModel.cpu()

    if not generate and opt.jamr == 0:
        if opt.train_posterior:
            for param in MrpModel.poserior_m.parameters():
                param.requires_grad = True
            MrpModel.poserior_m.apply(lambda x: freeze(x,opt.dropout))
        else:
            opt.prior_t = 0
            opt.sink_re = 0
            for param in MrpModel.poserior_m.parameters():
                param.requires_grad = False
    parameters_to_train = []
    if opt.retrain_all:
        for name, param in MrpModel.named_parameters():
            if name != "word_fix_lut":
                param.requires_grad = True
                parameters_to_train.append(param)
            else:
                logger.info("not updating "+name)

    else:
        if opt.rel:
            for param in MrpModel.concept_decoder.parameters():
                if   param.requires_grad:
                    param.requires_grad = False
                    logger.info("turing off concept model:  ",param)
            for name,p in MrpModel.named_parameters():
                if name == "word_fix_lut" or p.size(0) == len(dicts["word_dict"]):
                    p.requires_grad = False
                if p.requires_grad:
                    parameters_to_train.append(p)
        else:
            logger.info("paramters size:{}".format([p.size() for p in MrpModel.concept_decoder.parameters()]))
            MrpModel.apply(freeze)
            for p in MrpModel.concept_decoder.parameters():
                p.requires_grad = True
                parameters_to_train.append(p)
    logger.info("MrpModel:\n{}".format(MrpModel))
    logger.info("training parameters: {}".format(len(parameters_to_train)))
    return MrpModel,parameters_to_train,optt, optim
