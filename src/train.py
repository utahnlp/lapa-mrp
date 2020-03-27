#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to train the model

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from parser.Dict import *

from parser.DataIterator import *
from parser.BertDataIterator import *
import random
import json
from parser.models import *
import argparse
import logging
from src import *
from src.config_reader import *
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import PackedSequence
from torch import cuda
from pytorch_transformers.modeling_bert import BertModel
import time
from utility.amr_utils.AMRNaiveScores import *
from utility.dm_utils.DMNaiveScores import *
from utility.psd_utils.PSDNaiveScores import *
from torch.utils.tensorboard import SummaryWriter

def posterior_regularizor(posterior):
    '''probBatch:   tuple (src_len x  batch x n_out,lengths),
       tgtBatch: amr_len x batch x n_feature  , lengths
        posterior = packed( amr_len x  batch x src_len , lengths)

      total_loss,total_data
    '''

    assert isinstance(posterior, tuple),"only support tuple"
    # unpacked_posterior: [amr_len x batch x src_len], lengths, diffent amr_lens
    unpacked_posterior,lengths = unpack(posterior)
    activation =pack(unpacked_posterior.sum(0),lengths,batch_first=True)[0]
    activation_loss = torch.nn.functional.relu(activation-1).sum()
    return activation_loss

import math


def sinkhorn_score_regularizor(score):
    '''probBatch:   tuple (src_len x  batch x n_out,lengths),
        score = packed( data(amr) x src_len , amr_lengths)

      total_loss,total_data
    The KL part, see the supplementary material of ICLR https://arxiv.org/pdf/1802.08665.pdf
    LEARNING LATENT PERMUTATIONS WITH GUMBELSINKHORN NETWORKS
    '''

    scores,lengths = unpack(score)
    S = 0
    r= opt.prior_t/opt.sink_t
    gamma_r = math.gamma(1+r)
    for i,l in enumerate(lengths):
        # fix an error here, when scores is 0, it become inf
        # S = S + r / scores[:l,i,:l].sum()+gamma_r*torch.exp( -scores[:l,i,:l]*r).sum()
        # when scores is large negative, it become very large.
        S = S + r * scores[:l,i,:l].sum()+gamma_r*torch.exp( -scores[:l,i,:l]*r).sum()
    return S #+activation_loss

epsilon = 1e-6
import numpy as np
def Total_Loss(probs,tgtBatch,alginBatch,epoch,rel_roles_batch = None,roots_log_soft_max=None,gold_roots=None):
    '''probBatch:   tuple of packed((data(src_len), n_out),src_lengths) cat, le, ner
       tgtBatch: packed(data(amr) x n_feature)  , amr_lengths
       alginBatch: packed(data(amr_len) x src_len), amr_lengths
       posterior = packed(data(amr_len) x src_len , amr_lengths
       likeli = packed(data(amr)x src_len, amr_lengths
       score =  packed(data(amr)x src_len), amr_lengths
       roots = batch
       total_loss,total_data
    '''
    logger = logging.getLogger("mrp")
    probBatch,posteriors_likelihood_score = probs
    posterior,likeli,score = posteriors_likelihood_score
    final_device = likeli.data.device
    if  opt.jamr :
        # when it is jamr, use the alignment to get the likelihood P(c|w)
        # out : (amr_batch_len), src_len
        out = posterior.data*likeli.data
        # p : (amr_batch_len)
        p = out.sum(1)+epsilon
        total_loss = -( torch.log(p) ).sum()
    elif opt.renyi_alpha > 0:
        # soft loss
        out = alginBatch.data*(posterior.data+epsilon).pow(opt.renyi_alpha)*(likeli.data+epsilon).pow(1-opt.renyi_alpha)
        p = out.sum(1)+epsilon
        total_loss = torch.log(p).sum()/(opt.renyi_alpha-1)
    else: #hierachical relaxation
        out = alginBatch.data*posterior.data*likeli.data
        p = torch.nn.functional.relu(out.sum(1))+epsilon
        total_loss = -( torch.log(p) ).sum()
    # assert not np.isnan(total_loss.detach().cpu().numpy()).any(),("concept\n",epoch,posterior.data,likeli.data)

    # logger.info("alginBatch:{}, posterior:{}, likeli:{}, p:{}, total_loss:{}".format(alginBatch.data, posterior.data, likeli.data, p, total_loss))
    if opt.jamr  :
        posterior_loss = torch.zeros(1).to(device=final_device)
        sink_reg = torch.zeros(1).to(device=final_device)
    else:
        posterior_loss = opt.sink_re*posterior_regularizor(posterior) if  opt.sink_re != 0 else torch.zeros(1).to(device=final_device)
        if opt.prior_t :
            sink_reg = sinkhorn_score_regularizor(score)
        else:
            sink_reg = torch.zeros(1).to(device=final_device)

    final_posterior_loss = posterior_loss + sink_reg
    # assert not np.isnan(final_posterior_loss.detach().cpu().numpy()).any(), ("posterior\n",epoch,final_posterior_loss.detach())

    # logger.info("total_loss:{}, posterior_loss:{}, sink_reg:{}".format(total_loss, posterior_loss, sink_reg))
    total_data = out.size(0)
    if rel_roles_batch:
        _,rel_prob = probBatch
        root_loss = - sum(log_soft_max[i] for log_soft_max,i in zip(roots_log_soft_max,gold_roots))
        # assert not np.isnan(root_loss.detach().cpu().numpy()).any(), ("root_loss\n",epoch,root_loss.detach())
        total_roots =  len(gold_roots)  #- (rel_roles_batch.data.detach() == 0 ).sum()
        total_rel =  len(rel_roles_batch[0].detach())  #- (rel_roles_batch.data.data == 0 ).sum()
        #logger.info("rel_prob.data:{} \n size(rel_prob) = {}, rel_roles_batch: {}, max_min rel_roles_batch:{},{} )\n".format(rel_prob.data, rel_prob.data.size() , rel_roles_batch.data.cpu(), torch.max(rel_roles_batch.data), torch.min(rel_roles_batch.data)))
        rel_loss =  torch.nn.functional.nll_loss(rel_prob.data,rel_roles_batch.data, reduction='sum')
        # assert not np.isnan(rel_loss.detach().cpu().numpy()).any(), ("srlloss\n",epoch,rel_loss)
        packed_total_loss = ([total_loss,final_posterior_loss],root_loss, rel_loss)
        packed_total_data = (total_data, total_roots,total_rel )
        return packed_total_loss,packed_total_data
    else:
        return ([total_loss,final_posterior_loss]),total_data


def eval(model,decoder,scorer,data,dicts,epoch,rel=False):

    logger = logging.getLogger("mrp")
    # for evaluation, shuffle is not required
    data.shuffle()
    concept_scores = scorer.concept_score_initial(dicts)

    rel_scores = scorer.rel_scores_initial()

    # only set into eval model, do nothing on evaluation.
    model.eval()
    if opt.debug_size > 0:
        total_dev_batch = opt.debug_size
    else:
        total_dev_batch = len(data)

    for batchIdx in range(total_dev_batch):

        # use bert_model as a switch to feed bertBatch or not
        if opt.bert_model:
            order,idsBatch,srcBatch,src_charBatch, tgtBatch,alginBatch,rel_batch,rel_index_batch,rel_roles_batch,gold_roots,sourceBatch,srcBertBatch,srcBertIndexBatch = data[batchIdx]
        else:
            order,idsBatch,srcBatch,src_charBatch, tgtBatch,alginBatch,rel_batch,rel_index_batch,rel_roles_batch,gold_roots,sourceBatch = data[batchIdx]
            srcBertBatch = None
            srcBertIndexBatch = None

        # concept only evaluation
        probBatch, src_enc = model((srcBatch,src_charBatch), rel=False, bertBatch=srcBertBatch, bertIndexBatch = srcBertIndexBatch)

        # by default op.get_sense is true
        # here, the aligns is not aligned to the orignal token, but aligned to the categorized the nodes.
        # Another steps quired to convered into aligns to the original token, but when evaluation, tok -> categorized node is 1-to-1 mapping, they shared the same index, hence, here alings,can be directly used to retrieve the original tokens.
        concepts_pred_seq,concept_batches,aligns,dependent_mark_batch = decoder.probAndSourceToConcepts(sourceBatch,srcBatch,src_charBatch, probBatch,getsense= opt.get_sense and not rel)
        if rel:
            # and align batch here is also the fixed alignment after concept identification to the previous categorized alignment.
            # Another steps quired to convered into aligns to the original token, but when evaluation, tok -> categorized node is 1-to-1 mapping, they shared the same index, hence, here alings,can be directly used to retrieve the original tokens.
            rel_batch_data,align_batch = rel_to_batch(concept_batches,aligns,data,dicts,model.frame)

            rel_roles_prob,roots_log_soft_max = model((rel_batch_data,srcBatch, src_charBatch, src_enc, align_batch),rel=True,bertBatch=srcBertBatch,bertIndexBatch=srcBertIndexBatch )
            # by default, get wiki is false at this time.
            graphs,rel_triples  =  decoder.relProbAndConToGraph(concept_batches, sourceBatch, rel_roles_prob,roots_log_soft_max,(dependent_mark_batch,aligns),opt.get_sense,set_wiki=opt.get_wiki,normalizeMod=opt.normalize_mod)
            if opt.get_sense:
                concept_batches = decoder.graph_to_concepts_batches(graphs)
            for score_h in rel_scores:
                if score_h.second_filter:
                    t,p,tp = score_h.T_P_TP_Batch(rel_triples,list(zip(*sourceBatch))[R_IND_SOURCE_BATCH],second_filter_material = (concept_batches,list(zip(*sourceBatch))[C_IND_SOURCE_BATCH]))
                else:
                    t,p,tp = score_h.T_P_TP_Batch(rel_triples,list(zip(*sourceBatch))[R_IND_SOURCE_BATCH])

        # only log the first batch to check
        if batchIdx < 1:
            if rel:
                checkAndLogPred(sourceBatch, concepts_pred_seq, concept_batches, rel_triples, rel)
            else:
                checkAndLogConceptOnly(sourceBatch, concepts_pred_seq, concept_batches)

        for score_h in concept_scores:
            # scores are between uni
            # TODO: to compare the anchors, we should consider the anchors
            t,p,tp = score_h.T_P_TP_Batch(concept_batches,list(zip(*sourceBatch))[C_IND_SOURCE_BATCH])

    model.train()
    return concept_scores,rel_scores

def logScores(prefix_info, concept_scores, rel_scores, rel, show_error = math.inf):
    if rel:
        for score_h in concept_scores:
            logger.info("{}, Concept Scores: {}".format(prefix_info, score_h))
            if show_error != math.inf:
                score_h.show_error(t=show_error)
        for score_h in rel_scores:
            logger.info("{}, Rel Scores: {}".format(prefix_info, score_h))
            if show_error != math.inf:
                score_h.show_error(t=show_error)
    else:
        for score_h in concept_scores:
            logger.info("{}, Concept Scores: {}".format(prefix_info, score_h))
            if show_error != math.inf:
                score_h.show_error(t=show_error)

def checkAndLogConceptOnly(sourceBatch, concepts_pred_seq, concept_batches):
    id = random.randrange(len(sourceBatch))
    logger.info("""Check concepts when evaluating:\n
    \t source_tokens: {0}\n
    \t pred_concepts: {1}\n
    \t final_pred_concepts: {2}\n
    \t gold_reCat_concepts: {3}\n
    \t gold_concepts: {4}\n\n""".format(
        sourceBatch[id][:TOTAL_INPUT_SOURCE],
        concepts_pred_seq[id],
        concept_batches[id],
        sourceBatch[id][CN_IND_SOURCE_BATCH],
        sourceBatch[id][C_IND_SOURCE_BATCH]
    ))
    return id

def checkAndLogPred(sourceBatch, concepts_pred_seq, concept_batches, rel_triples, rel):
    id = checkAndLogConceptOnly(sourceBatch, concepts_pred_seq, concept_batches)

    if rel:
        logger.info("""Check relations when evaluating:\n
        \t pred_triples: {0}\n
        \t gold_triples: {1}\n\n""".format(
        sorted([quandraple[:-2] for quandraple in rel_triples[id]], key=lambda x: (x[0].le, x[2])),
        sorted(sourceBatch[id][R_IND_SOURCE_BATCH], key=lambda x: (x[0].le, x[2]))
    ))

def trainModel(model, decoder, scorer, trainData, validData, dicts, optim,best_f1 = 0 ,best_epoch = 0):
    """
    train a model for different parser
    TODO: we need a unified represenation for graph parsing.
    """
    logger = logging.getLogger("mrp")
    logger.info(' * number of training sentences. {}'.format(len(trainData.src)))
    model.train()
    decoder.train()
    last_f1 = 0
    badstrike_epochs = 0
    start_time = time.time()
    def trainEpoch(epoch):
        total_loss, report_loss = 0, 0
        posterior_total_loss, posterior_report_loss = 0, 0
        total_words, report_words = 0, 0

        root_total_loss, root_report_loss = 0, 0
        root_total_words, root_report_words = 0, 0

        rel_total_loss, rel_report_loss = 0, 0
        rel_total_words, rel_report_words = 0, 0


        start = time.time()

        # i is ith batch in trainData
        if opt.debug_size > 0:
            total_batch = opt.debug_size
        else:
            total_batch = len(trainData)

        for i in range(total_batch):
            if opt.bert_model:
                order,idsBatch, srcBatch,src_charBatch,tgtBatch,alginBatch,rel_batch,rel_index_batch,rel_roles_batch,gold_roots,sourceBatch,srcBertBatch,srcBertIndexBatch = trainData[i]
            else:
                order,idsBatch, srcBatch,src_charBatch, tgtBatch,alginBatch,rel_batch,rel_index_batch,rel_roles_batch,gold_roots,sourceBatch = trainData[i]
                srcBertBatch = None
                srcBertIndexBatch = None

            # only for concept
            # srcBatch: packed[packed_batch_src_len x src_feature_dim]
            # tgtBatch: packed[packed_batch_re_amr_len x tgt_feature_dim]
            # alginbatch: packed[packed_batch_re_amr_len x src_len]
            # srcBertBatch: tensor [batch x bert_max_length]
            # probBatch: PackedSequence(cat_prob,batch_sizes),PackedSequence(le_prob,batch_sizes),PackedSequence(ner_prob,batch_sizes), all is packed_src_len x dim
            # posterior_likelihoood_score: posterior, likelihood, score
            # posterior: packed_batch_re_amr_len x src_len, batch_sizes
            # likelihood: packed_batch_re_amr_len x src_len, batch_sizes
            # score: packed_batch_re_amr_len x src_len, batch_sizes
            probBatch,posteriors_likelihood_score,src_enc = model((srcBatch,src_charBatch,tgtBatch,alginBatch ),rel=False, bertBatch=srcBertBatch, bertIndexBatch=srcBertIndexBatch)

            # for relation training
            if opt.rel:
                    # rel_batch:  mypacked_seq[packed_batch_gold_amr_len x tgt_feature_dim]
                    # rel_index_batch:  list(batch, real_gold_amr_len), but content is the index of recatogrized amr index, is a mapping
                    # srcBatch: packed[packed_batch_src_len x src_feature_dim]
                    # srcEnc : packed [packed_batch_src_len x src_enc_dim]
                    # posterior: packed [packed_batch_re_amr_len x src_len, batch_sizes]
                    # srcBertBatch: tensor [batch x bert_max_length]
                rel_prob,roots = model((rel_batch,rel_index_batch,srcBatch, src_charBatch, src_enc, posteriors_likelihood_score[0]),rel=True, bertBatch=srcBertBatch, bertIndexBatch=srcBertIndexBatch)
                out =  (probBatch,rel_prob),posteriors_likelihood_score
                packed_total_loss,packed_num_data = Total_Loss(out,tgtBatch,alginBatch,epoch,rel_roles_batch,roots,gold_roots)
            else:
                # when no rel, it concept only model
                out = probBatch,posteriors_likelihood_score
                packed_total_loss,packed_num_data = Total_Loss(out,tgtBatch,alginBatch,epoch)


            # rel loss
            if len(packed_total_loss)>2:
                packed_concept_loss,root_loss,rel_loss = packed_total_loss
                num_data,num_root,num_rel = packed_num_data

                rel_report_loss += rel_loss.item()
                rel_total_loss += rel_loss.item()

                rel_total_words += num_rel
                rel_report_words += num_rel

                root_total_loss += root_loss.item()
                root_report_loss += root_loss.item()

                root_total_words += num_root

                root_report_words += num_root
            else:
                packed_concept_loss = packed_total_loss
                num_data = packed_num_data

            epoch_batch_info = "Epoch " + str(epoch)+" "+ str(i) +"//"+ str(total_batch)
            # concept loss
            concept_total_loss,posterior_loss = packed_concept_loss
            all_loss = concept_total_loss + posterior_loss
            report_loss += all_loss.item()
            total_loss += all_loss.item()
            if opt.prior_t:
                posterior_report_loss+= posterior_loss.item()
                posterior_total_loss += posterior_loss.item()

            total_words += num_data
            report_words += num_data

            # TODO: support multiple GPU

            # cal rel loss
            if opt.rel:
                avg_loss = all_loss/num_data+root_loss/num_root + opt.rel*rel_loss/num_data#*epoch/opt.epochs
            else:
                # cal concept loss
                avg_loss = all_loss/num_data

            if concept_total_loss.item() > 1e7 or posterior_loss.item() > 1e7:
                logger.warn("{0}, concept_loss={1}, posterior_loss={2}".format(epoch_batch_info, concept_total_loss, posterior_loss))
           # only accumulation, no update yet
            avg_loss.backward()
            if ((i +  1) % opt.gradient_accumulation_steps == 0) or (i == total_batch - 1):
                #old_state_dict = {}
                #for key in model.state_dict(keep_vars=True):
                #    old_state_dict[key] = model.state_dict()[key].clone()
                grad_norm = optim.step()
                # clear the gradients
                optim.optimizer.zero_grad()
                logger.info("{0}, global_step: {1}, current grad_norm: {2:.4f}".format(epoch_batch_info, optim.get_est_global_step(), grad_norm))
                #new_state_dict = {}
                #for key in model.state_dict(keep_vars=True):
                #    new_state_dict[key] = model.state_dict()[key].clone()

                ## Compare params
                #count = 0
                #for key in old_state_dict:
                #    if not (old_state_dict[key] == new_state_dict[key]).all():
                #        count = count + 1
                #        # logger.info('Diff in {}, Para: {}'.format(key, new_state_dict[key]))
                #        logger.info('Diff in {}, Para_size: {}'.format(key, new_state_dict[key].size()))
                #logger.info("updates/total: {}/{}".format(count, len(old_state_dict.keys())))

                # here is log how many times in an epoch
            #if True:
            if i % (1+ int(total_batch/opt.log_per_epoch)) == 0 and i > 0:
                if opt.rel:
                    logger.info("{0}, current batch avg loss: {1}, concept_loss={2}, posterior_loss={3}, num_data = {4}, root_loss={5}, num_roots={6}, rel_loss:{7}".format(epoch_batch_info, avg_loss,concept_total_loss/num_data,posterior_loss/num_data, num_data, root_loss/num_root, num_root, rel_loss/num_data))
                else:
                    logger.info("{0}, current batch avg loss: {1}, concept_loss={2}, posterior_loss={3}, num_data = {4}".format(epoch_batch_info, avg_loss,concept_total_loss/num_data,posterior_loss/num_data, num_data))

                logger.info("{0}, concept loss: {1:.4f}".format(epoch_batch_info, 1.0*report_loss/report_words))
                if opt.prior_t:
                   logger.info("{0}, posterior loss: {1:.4f}".format(epoch_batch_info, posterior_report_loss/report_words))
                if rel_report_words > 0 :
                    if rel_report_loss/rel_report_words<50:
                        logger.info("{0}, rel perplexity: {1:.4f}".format(epoch_batch_info, math.exp(1.0*rel_report_loss/rel_report_words)))
                    else:
                        logger.info("{0}, rel loss: {1:.4f}".format(epoch_batch_info, rel_report_loss/rel_report_words))

                if root_report_words > 0 :
                    if root_report_loss/root_report_words<50:
                        logger.info("{0}, root  perplexity: {1:.4f}".format(epoch_batch_info, math.exp(1.0*root_report_loss/root_report_words)))
                    else:
                        logger.info("{0}, root loss: {1:.4f}".format(epoch_batch_info, root_report_loss/root_report_words))

                logger.info("{0}, tokens/s: {1:.4f}, {2:.4f} elapsed".format(epoch_batch_info, 1.0*report_words/(time.time()-start), time.time()-start_time))

                report_loss = report_words = 0
                rel_report_loss = rel_report_words = 0
                arg_rel_report_loss = arg_rel_report_words = 0

                start = time.time()
                #with torch.no_grad():
                #    concept_scores,rel_scores= eval(model,decoder,scorer, validData,dicts, epoch, rel=opt.rel)
                #    #concept_scores,rel_scores= eval(model,decoder,scorer, trainData,dicts, epoch, rel=opt.rel)
                #    logScores(epoch_batch_info, concept_scores, rel_scores, rel=opt.rel)

        return ((total_loss / total_words,posterior_total_loss/total_words),root_total_loss/root_total_words if root_total_words > 0 else 0,rel_total_loss/rel_total_words if rel_total_words > 0 else 0)

    for epoch in range(opt.start_epoch, opt.epochs + 1):

        epoch_summary_prefix = "Summary-Epoch {}".format(epoch)

        if epoch % 10 == 0:
            logger.info("{}, Options: {}".format(epoch_summary_prefix, opt))
 
        for name, param in model.named_parameters():
            assert not np.isnan(np.sum(param.data.cpu().numpy())),(name,"\n",param)
        #  (1) train for one epoch on the training set
        if opt.debug_size <= 0:
            trainData.shuffle()
        train_loss = trainEpoch(epoch)
        # train_loss = ((0.1, 0.2),0.3, 0.4)

        concept_loss,posterior_loss = train_loss[0]
        logger.info('{}, Concept loss: {}'.format(epoch_summary_prefix, concept_loss))
        if opt.prior_t:
            logger.info('{}, Posterior_ppl loss: {}'.format(epoch_summary_prefix, posterior_loss))

        if train_loss[1] != 0:
            root_train_ppl = math.exp(train_loss[1]) if train_loss[1]<50 else math.nan
            logger.info('{}, Root_ppl loss: {}'.format(epoch_summary_prefix, root_train_ppl))

            role_train_ppl = math.exp(train_loss[2]) if train_loss[2]<50 else math.nan
            logger.info('{}, Role_ppl loss: {}'.format(epoch_summary_prefix, role_train_ppl))

        #  (2) maybe update the learning rate, according to epoch
        if opt.optim == 'sgd':
            optim.updateLearningRateWithLRDecay(concept_loss, epoch)

        #  (3) evaluate on the validation set
        with torch.no_grad():
            if opt.debug_size > 0:
                # evaluation on training
                concept_scores,rel_scores= eval(model,decoder,scorer,trainData,dicts, epoch, rel=opt.rel)
            else:
                concept_scores,rel_scores= eval(model,decoder,scorer,validData,dicts,epoch, rel=opt.rel)
            logScores(epoch_summary_prefix, concept_scores, rel_scores, rel=opt.rel, show_error = 10 if epoch % 5==1 else math.inf)
            p,r,f1 = scorer.get_smatch(concept_scores, rel_scores, rel = opt.rel)

            if f1 > best_f1:
                #  (4) drop a checkpoint
                checkpoint = {
                    'model': model.state_dict(),
                    'opt': opt,
                    'epoch': epoch,
                    'optim': optim.optimizer.state_dict()
                }
                best_f1 = f1
                last_f1 = f1
                badstrike_epochs = 0
                best_epoch = epoch
                logger.info("{}, saving best model in {}".format(epoch_summary_prefix, opt.save_to+opt_str+'valid_best.pt'))
                torch.save(checkpoint, opt.save_to+opt_str+'valid_best.pt' )
            else:
                if f1 <= last_f1:
                    badstrike_epochs = badstrike_epochs + 1
                last_f1 = f1

            logger.info('{}, Validation F1: {}'.format(epoch_summary_prefix, f1))
            logger.info('{}, Best Validation F1: {}'.format(epoch_summary_prefix, best_f1))
            logger.info("{}, Best Epoch so on: {}".format(epoch_summary_prefix, best_epoch))

            if badstrike_epochs > 5:
                logger.warn("{}, badstrike_epochs: {}, stop training".format(epoch_summary_prefix, badstrike_epochs))
                break

    # after all training epoch, save state_dict is much safe than safe than entire model
    # lambda function cannot be serialized by pickle.
    checkpoint = {
        'model': model.state_dict(),
        'opt': opt,
        'epoch': epoch,
        'optim': optim.optimizer.state_dict()
    }

    logger.info("\nEvaluate Training After {} epochs".format(epoch))

    with torch.no_grad():
        concept_scores,rel_scores= eval(model,decoder,scorer, trainData,dicts, epoch, rel=opt.rel)
        logScores(epoch, concept_scores, rel_scores, rel = opt.rel, show_error=20)
        p_r_f1 = scorer.get_smatch(concept_scores, rel_scores, rel = opt.rel)

        logger.info("After {} epochs, Training Total Precesion, recall, f1, {}".format(epoch, p_r_f1))
    logger.info("After {} epochs, Saving last model".format(epoch))
    torch.save(checkpoint, opt.save_to+opt_str+'last.pt' )



def embedding_from_dicts( opt,dicts):

    logger = logging.getLogger("mrp")
    embs = dict()

    def add_amr_embeddings():
        amr_high_lut = nn.Embedding(dicts["amr_high_dict"].size(), opt.lemma_dim, padding_idx=PAD)
        logger.info("amr_high_lut {}, amr_high_dict:\n {}".format(amr_high_lut.num_embeddings, str(dicts["amr_high_dict"])))

        amr_rel_lut =  nn.Embedding(dicts["amr_rel_dict"].size(), 1)  #not actually used, but handy to pass number of relations
        logger.info("amr_rel_lut {}, amr_rel_dict:\n {} ".format(amr_rel_lut.num_embeddings, str(dicts["amr_rel_dict"])))

        amr_cat_lut = nn.Embedding(dicts["amr_category_dict"].size(), opt.cat_dim, padding_idx=PAD)

        logger.info("amr_cat_lut {}, amr_cat_dict:\n {}".format(amr_cat_lut.num_embeddings, str(dicts["amr_category_dict"])))

        amr_aux_lut = nn.Embedding(dicts["amr_aux_dict"].size(), 1, padding_idx=PAD)
        logger.info("amr_aux_lut {}, amr_aux_dict:\n {}".format(amr_aux_lut.num_embeddings, str(dicts["amr_aux_dict"])))

        if opt.cuda:
            amr_rel_lut.cuda()
            amr_cat_lut.cuda()
            amr_aux_lut.cuda()
            amr_high_lut.cuda()

        embs["amr_aux_lut"] = amr_aux_lut
        embs["amr_high_lut"] = amr_high_lut
        embs["amr_cat_lut"] = amr_cat_lut
        embs["amr_rel_lut"] = amr_rel_lut


    def add_dm_embeddings():
        dm_target_pos_lut = nn.Embedding(dicts["dm_target_pos_dict"].size(), opt.pos_dim, padding_idx=PAD)
        logger.info("dm_target_pos_lut {}, dm_target_pos_lu :\n {}".format(dm_target_pos_lut.num_embeddings, str(dicts["dm_target_pos_dict"])))

        dm_rel_lut =  nn.Embedding(dicts["dm_rel_dict"].size(), 1)  #not actually used, but handy to pass number of relations
        logger.info("dm_rel_lut {}, dm_rel_dict:\n {} ".format(dm_rel_lut.num_embeddings, str(dicts["dm_rel_dict"])))

        dm_cat_lut = nn.Embedding(dicts["dm_cat_dict"].size(), opt.lemma_dim, padding_idx=PAD)
        logger.info("dm_cat_lut {}, dm_cat_dict:\n {}".format(dm_cat_lut.num_embeddings, str(dicts["dm_cat_dict"])))

        dm_high_lut = nn.Embedding(dicts["dm_high_dict"].size(), opt.cat_dim, padding_idx=PAD)
        logger.info("dm_high_lut {}, dm_high_dict:\n {}".format(dm_high_lut.num_embeddings, str(dicts["dm_high_dict"])))
        #dm_high_le_lut = nn.Embedding(dicts["dm_high_le_dict"].size(), opt.lemma_dim, padding_idx=PAD)
        #logger.info("dm_high_le_lut {}, dm_high_le_dict:\n {}".format(dm_high_le_lut.num_embeddings, str(dicts["dm_high_le_dict"])))
        dm_sense_lut = nn.Embedding(dicts["dm_sense_dict"].size(), 1, padding_idx=PAD)
        logger.info("dm_sense_lut {}, dm_sense_dict:\n {}".format(dm_sense_lut.num_embeddings, str(dicts["dm_sense_dict"])))

        if opt.cuda:
            dm_target_pos_lut.cuda()
            dm_cat_lut.cuda()
            dm_sense_lut.cuda()
            dm_rel_lut.cuda()
            dm_high_lut.cuda()

        embs["dm_target_pos_lut"] = dm_target_pos_lut
        embs["dm_cat_lut"] = dm_cat_lut
        embs["dm_sense_lut"] = dm_sense_lut
        embs["dm_rel_lut"] = dm_rel_lut
        embs["dm_high_lut"] = dm_high_lut
        #embs["dm_high_le_lut"] = dm_high_le_lut

    def add_psd_embeddings():
        """
        add psd specific embedding
        """
        psd_target_pos_lut = nn.Embedding(dicts["psd_target_pos_dict"].size(), opt.pos_dim, padding_idx=PAD)
        logger.info("psd_target_pos_lut {}, psd_target_pos_lu :\n {}".format(psd_target_pos_lut.num_embeddings, str(dicts["psd_target_pos_dict"])))

        psd_rel_lut =  nn.Embedding(dicts["psd_rel_dict"].size(), 1, padding_idx=PAD)  #not actually used, but handy to pass number of relations
        logger.info("psd_rel_lut {}, psd_rel_dict:\n {} ".format(psd_rel_lut.num_embeddings, str(dicts["psd_rel_dict"])))

        psd_sense_lut = nn.Embedding(dicts["psd_sense_dict"].size(), 1, padding_idx=PAD)
        logger.info("psd_sense_lut {}, psd_sense_dict:\n {}".format(psd_sense_lut.num_embeddings, str(dicts["psd_sense_dict"])))

        psd_high_lut = nn.Embedding(dicts["psd_high_dict"].size(), opt.lemma_dim, padding_idx=PAD)
        logger.info("psd_high_lut {}, psd_high_dict:\n {}".format(psd_high_lut.num_embeddings, str(dicts["psd_high_dict"])))

        if opt.cuda:
            psd_target_pos_lut.cuda()
            psd_sense_lut.cuda()
            psd_rel_lut.cuda()
            psd_high_lut.cuda()

        embs["psd_target_pos_lut"] = psd_target_pos_lut
        embs["psd_sense_lut"] = psd_sense_lut
        embs["psd_rel_lut"] = psd_rel_lut
        embs["psd_high_lut"] = psd_high_lut


    def add_eds_embeddings():
        """
        adding eds specific embedding
        """
        pass

    def add_ucca_embeddings():
        """
        add ucca specifice  emebedding
        """
        pass

    def initial_embedding():
        logger.info("loading fixed word embedding")

        word_dict = dicts["word_dict"]
        lemma_dict = dicts["lemma_dict"]
        word_initialized = 0
        lemma_initialized = 0
        word_embedding = nn.Embedding(dicts["word_dict"].size(),
                                      300,   #size of glove dimension
                              padding_idx=PAD)
        lemma_lut = nn.Embedding(dicts["lemma_dict"].size(),
                                  opt.lemma_dim,
                                  padding_idx=PAD)
        with open(embed_path, 'r') as f:
            for line in f:
                parts = line.rstrip().split()
                id,id2 = word_dict[parts[0]],lemma_dict[parts[0]]
                if id != UNK and id < word_embedding.num_embeddings:
                    tensor = torch.FloatTensor([float(s) for s in parts[-word_embedding.embedding_dim:]]).type_as(word_embedding.weight.data)
                    word_embedding.weight.data[id].copy_(tensor)
                    word_initialized += 1

                # didn't copy data
                if False and id2 != UNK and id2 < lemma_lut.num_embeddings :
                    tensor = torch.FloatTensor([float(s) for s in parts[-lemma_lut.embedding_dim:]]).type_as(lemma_lut.weight.data)
                    lemma_lut.weight.data[id2].copy_(tensor)
                    lemma_initialized += 1

        logger.info("word_initialized {}".format(word_initialized))
        logger.info("lemma initialized {}".format(lemma_initialized))
        logger.info("word_total {}".format(word_embedding.num_embeddings))
        return word_embedding,lemma_lut
    
    def add_common_embeddings():
        if opt.initialize_word:
            word_fix_lut,lemma_lut = initial_embedding()
            word_fix_lut.requires_grad = False      ##need load
        else:
            word_fix_lut =  nn.Embedding(dicts["word_dict"].size(), 300, padding_idx=PAD)
            lemma_lut = nn.Embedding(dicts["lemma_dict"].size(), opt.lemma_dim, padding_idx=PAD)

        char_lut = nn.Embedding(dicts["char_dict"].size(), opt.char_dim, padding_idx=PAD)
        logger.info("char_lut {}, char_dict:\n {}".format(char_lut.num_embeddings, str(dicts["char_dict"])))

        pos_lut = nn.Embedding(dicts["pos_dict"].size(), opt.pos_dim, padding_idx=PAD)
        logger.info("pos_lut {}, pos_dict:\n {}".format(pos_lut.num_embeddings, str(dicts["pos_dict"])))

        ner_lut = nn.Embedding(dicts["ner_dict"].size(), opt.ner_dim, padding_idx=PAD)
        logger.info("ner_lut {}, ner_dict:\n {} ".format(ner_lut.num_embeddings, str(dicts["ner_dict"])))

        word_fix_lut.cpu()
        if opt.cuda:
            lemma_lut.cuda()
            pos_lut.cuda()
            char_lut.cuda()
            ner_lut.cuda()

        embs["word_fix_lut"] = word_fix_lut
        embs["lemma_lut"] = lemma_lut
        embs["pos_lut"] = pos_lut
        embs["char_lut"] = char_lut
        embs["ner_lut"] = ner_lut


    add_common_embeddings()
    for frame in opt.frames:
        if frame == "amr":
            add_amr_embeddings()

        if frame == "dm":
            add_dm_embeddings()

        if frame == "psd":
            add_psd_embeddings()

        if frame == "eds":
            add_eds_embeddings()

        if frame == "ucca":
            add_ucca_embeddings()
    return embs

def create_mrp_decoder(frame, dicts):
    if frame == "amr":
        decoder = parser.AMRProcessors.AMRDecoder(opt,dicts)
    elif frame == "dm":
        decoder = parser.DMProcessors.DMDecoder(opt,dicts)
    elif frame == "psd":
        decoder = parser.PSDProcessors.PSDDecoder(opt,dicts)
    else:
        raise NotImplementedError("{} decoder is not supported".format(frame))

    return decoder

def create_mrp_dataiterators(frame, dicts):
    if frame == "amr":
        with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
        # take preprocessed pickle based file as input
        suffix = "." + frame+ "_pickle"+with_jamr+"_processed"
        rel_dict = dicts["amr_rel_dict"]
    else:
        suffix = "."+ frame + "_pickle_processed"
        rel_dict = dicts["{}_rel_dict".format(frame)]

    trainFolderPath = opt.build_folder+"/training/"
    trainingFilesPath = folder_to_files_path(trainFolderPath,suffix)

    devFolderPath = opt.build_folder +"/dev/"
    devFilesPath = folder_to_files_path(devFolderPath,suffix)

    if opt.bert_model:
        dev_data = BertDataIterator(devFilesPath,opt,rel_dict)
        training_data = BertDataIterator(trainingFilesPath,opt,rel_dict)
    else:
        dev_data = DataIterator(devFilesPath,opt,rel_dict)
        training_data = DataIterator(trainingFilesPath,opt,rel_dict)

    return {"training": training_data, "dev": dev_data}


def create_mrp_model(frame, dicts, embs, component_dict, decoder, scorer, data_iterators = None):
    f1 = 0
    if not opt.restore_from:
        model = parser.models.MRPModel(opt, embs, component_dict=component_dict, frame=frame)
        return model, f1, None
    else:
        model,parameters_to_train,optt, optimState = load_old_model(frame, dicts,opt)
        opt.start_epoch =  1

        # reloaded model, evaluate score on concepts and rels
        with torch.no_grad():
            concept_scores,rel_scores= eval(model, decoder, scorer, data_iterators["dev"], dicts, epoch = 0, rel=False)
            logScores("Reload {}".format(opt.restore_from), concept_scores, rel_scores, rel=opt.rel, show_error = 20)
            p, r, f1 = scorer.get_smatch(concept_scores, rel_scores, rel = opt.rel)
            logger.info("Reload {}, best_f1 {}".format(opt.restore_from, f1))
        return model, f1, optimState

def create_mrp_scorer(frame):
    if frame == "amr":
        return AMRNaiveScores()
    elif frame == "dm":
        return DMNaiveScores()
    elif frame == "psd":
        return PSDNaiveScores()
    else:
        raise NotImplementedError("{} is not supported".format(frame))

def main():
    dicts = read_all_dicts(opt.build_folder, opt.frames)
    embs = embedding_from_dicts( opt,dicts)
    logger = logging.getLogger("mrp")
    logger.info('Building model...')

    decoder_dict = {}
    model_dict = {}
    current_f1_dict = {}
    data_iterator_dict = {}
    component_dict = {}
    scorer_dict = {}

    # TODO: to support all
    for frame in opt.frames:
        decoder_dict[frame] = create_mrp_decoder(frame, dicts)
        data_iterator_dict[frame] = create_mrp_dataiterators(frame, dicts)
        scorer_dict[frame] = create_mrp_scorer(frame)
        model, f1, optimOption = create_mrp_model(frame, dicts, embs, component_dict, decoder_dict[frame], scorer_dict[frame], data_iterator_dict[frame])
        model_dict[frame] = model
        current_f1_dict[frame] = f1

    mtlModel = MTLModel(model_dict)
    parameters_to_train = []
    for p in mtlModel.parameters():
        if p.requires_grad:
            parameters_to_train.append(p)
    # training related options, the total steps to use for training.
    if opt.debug_size > 0:
        opt.num_train_optimization_steps = int(opt.debug_size / opt.gradient_accumulation_steps) * opt.epochs
    else:
        max_training_data = 0
        for frame in opt.frames:
            x = len(data_iterator_dict[frame]['training'])
            logger.info("frame_{} = {}".format(frame, x))
            if x > max_training_data:
                max_training_data = x
        opt.num_train_optimization_steps = int(max_training_data / opt.gradient_accumulation_steps) * opt.epochs
        logger.info("max_training_dara{} / gradient_accu{} * opt.epochs{} = {}, {}".format(max_training_data, opt.gradient_accumulation_steps, opt.epochs, int(max_training_data / opt.gradient_accumulation_steps), opt.num_train_optimization_steps))

    if opt.optim_json_configs:
        opt.optim_dict_configs = json.loads(opt.optim_json_configs)
        logger.info("load grouped_optim_configs into dict {}".format(opt.optim_dict_configs))
    else:
        opt.optim_dict_configs = None

    optim = parser.Optim.Optim(
        # parameters_to_train, opt.optim, opt.learning_rate, opt.max_grad_norm,
        mtlModel.named_parameters(), opt.optim, opt.learning_rate, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        weight_decay=opt.weight_decay,
        warmup_proportion=opt.warmup_proportion,
        num_train_optimization_steps = opt.num_train_optimization_steps,
        optim_scheduler_name = opt.optim_scheduler_name,
        grouped_dict_configs = opt.optim_dict_configs
    )

    logger.info("Optimizer Initialized: {}".format(optim.optimizer))
    logger.info("Whole model is as follows:\n {}".format(mtlModel))
    logger.info(' * batch size. {}'.format(opt.batch_size))

    #if optimOption != None:
    #    optim.optimizer.load_state_dict(optimOption)
    #    logger.info("Optimizer resume previous state: {}".format(optim.optimizer))
    nParams = sum([p.nelement() for p in mtlModel.parameters()])
    logger.info(' * number of parameters: {}'.format(nParams))

    trainable_nParams = sum([p.nelement() for p in parameters_to_train])
    logger.info(' * number of trainable parameters: {}'.format(trainable_nParams))

    #with torch.autograd.detect_anomaly():
    # TODO: now only used one decoder, later, we can try multple decoders
    for frame in opt.frames:
        trainModel(model_dict[frame], decoder_dict[frame], scorer_dict[frame], data_iterator_dict[frame]["training"], data_iterator_dict[frame]["dev"], dicts, optim, best_f1=current_f1_dict[frame])

if __name__ == "__main__":
    global opt
    global opt_str
    global sum_writer
    torch.set_printoptions(threshold=10000)
    opt = get_parser().parse_args()
    if opt.summary_dir:
        sum_writer = SummaryWriter(opt.summary_dir)
    else:
        sum_writer = None

    logger = logging.getLogger("mrp")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if opt.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info("Logger inilitialized.")
    with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"

    opt.lemma_dim = opt.dim
    opt.high_dim = opt.dim

    #used for deciding saved model name, gpus default is 0
    if opt.gpus[0] == -1:
        # cpu
        gpus_str = "-1"
        opt.cuda = 0
        logger.warn("Attention Please!!! Run on cpus:{}".format(gpus_str))
    elif opt.gpus[0] == -2:
        # trust the CUDA_VISIBLE_DEVICE
        # schedule by the system automatically, trust the CUDA_VISIBLE_DEVICES
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            gpus_str = os.environ["CUDA_VISIBLE_DEVICES"]
            opt.cuda = len(gpus_str.split(','))
            logger.info("Trust the scheduler, which set the CUDA_VISIBLE_DEVICE to {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        else:
            # default will use the first one, it is 0
            gpus_str = "0"
            opt.cuda = 1
            logger.info("No CUDA_VISIBLE_DEVICE found, use default gpu:0, avaible gpu counts: {}".format(torch.cuda.device_count()))
    else:
        # when speficy the gpus to use.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in opt.gpus])
        gpus_str = os.environ['CUDA_VISIBLE_DEVICES']
        opt.cuda = len(opt.gpus)
        logger.info("CUDA_VISIBLE_DEVICES speificed with {}, avauble cuda device_count".format(os.environ["CUDA_VISIBLE_DEVICES"], torch.cuda.device_count()))

    opt_str =  "gpus_"+ gpus_str
    logger.info("model prefix opt_str: {}".format(opt_str))
    logger.info("parsed options are {}".format(opt))

    if torch.cuda.is_available() and not opt.cuda:
        logger.warn("Attention: You have a CUDA device, so you should probably run with -cuda")

    # options for training
    if opt.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            opt.gradient_accumulation_steps))

    # split batchsize into sererval steps
    opt.batch_size = opt.batch_size // opt.gradient_accumulation_steps

    # fix seeds for initial comparing
    torch.manual_seed(0)
    if opt.cuda > 0:
        torch.cuda.manual_seed_all(0)
        # reset gpu memory from last exceptional run
        torch.cuda.empty_cache()
        # improve the speed in benchmark model, but slightly randomness when feedforward, used together with determistic
        torch.backends.cudnn.benchmark = True
        # make it determistic
        torch.backends.cudnn.deterministic = True

    ## it is used to specify the gpus to use
    ## it is not recommended for using this.
    #if opt.cuda:
    #    cuda.set_device(opt.gpus[0])
    main()
