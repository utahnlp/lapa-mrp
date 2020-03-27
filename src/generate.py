#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to run the model over preprocessed data to generate evaluatable results

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from parser.DataIterator import *
from parser.BertDataIterator import *

import parser
import torch
import os
import networkx as nx
from torch import cuda
from utility.amr_utils.AMRNaiveScores import *
from utility.dm_utils.DMNaiveScores import *
from parser.AMRProcessors import *
from parser.DMProcessors import *
from utility.data_helper import folder_to_files_path
from src import *
from parser.Dict import read_dicts
from src.config_reader import get_parser

logger = logging.getLogger("mrp")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


frame2flavors_mapping = {
    "amr" : 2,
    "dm"  : 0,
    "psd" : 0,
    "eds" : 1,
    "ucca" :1
}

def generate_parser():
    parser = get_parser()
    parser.add_argument('-output', default="_generate", help="""the suffix added for generated output file""")
    parser.add_argument('-with_graphs', type=int,default=1)
    return parser

def get_outfile(input_file, output_folder, output_suffix):
    input_file = input_file + "_txt" + output_suffix
    return output_folder + os.path.basename(input_file)

def get_mrp_outfile(input_file, output_folder, output_suffix):
    input_file = input_file + "_mrp" + output_suffix
    return output_folder + os.path.basename(input_file)

def generate_graph(set_name, model,decoder,scorer, data_set,dicts,file):

    concept_scores = scorer.concept_score_initial(dicts)

    rel_scores = scorer.rel_scores_initial()

    input_dict = {}
    with open(file, "r") as snt_file:
        for graph,_ in mrp_read(snt_file):
            input_dict[graph.id] = graph

    model.eval()
    decoder.eval()
    output = []
    gold_file = []
    mrp_output = []
    for batchIdx in range(len(data_set)):
        if opt.bert_model:
            if set_name != "test":
                order,idsBatch,srcBatch, src_charBatch,_,_,_,_,_,gold_roots,sourceBatch,srcBertBatch, srcBertIndexBatch =data_set[batchIdx]
            else:
                order,idsBatch,srcBatch,src_charBatch,sourceBatch,srcBertBatch, srcBertIndexBatch =data_set[batchIdx]
        else:
            if set_name != "test":
                order,idsBatch,srcBatch, src_charBatch,_,_,_,_,_,gold_roots,sourceBatch =data_set[batchIdx]
            else:
                order,idsBatch,srcBatch, src_charBatch,sourceBatch =data_set[batchIdx]
            srcBertBatch = None
            srcBertIndexBatch = None

        probBatch, src_enc = model((srcBatch,src_charBatch), rel=False, bertBatch = srcBertBatch, bertIndexBatch=srcBertIndexBatch)

        concept_pred_seq,concept_batches,aligns_raw,dependent_mark_batch = decoder.probAndSourceToConcepts(sourceBatch,srcBatch,src_charBatch,probBatch,getsense = opt.get_sense )

        concept_pred_seq = [ [uni.to_tuple() for uni in seq ] for  seq in concept_pred_seq ]

        rel_batch,aligns = rel_to_batch(concept_batches,aligns_raw,data_set,dicts, decoder.frame)
        rel_prob,roots = model((rel_batch, srcBatch, src_charBatch, src_enc, aligns),rel=True,bertBatch=srcBertBatch, bertIndexBatch=srcBertIndexBatch)
        graphs,rel_triples  =  decoder.relProbAndConToGraph(concept_batches,sourceBatch, rel_prob,roots,(dependent_mark_batch,aligns),opt.get_sense,set_wiki=opt.get_wiki,normalizeMod=opt.normalize_mod)
        concept_batches = decoder.graph_to_concepts_batches(graphs)
        batch_out = [0]*len(graphs)
        if set_name != "test":
            for score_h in rel_scores:
                if score_h.second_filter:
                    t,p,tp = score_h.T_P_TP_Batch(rel_triples,list(zip(*sourceBatch))[R_IND_SOURCE_BATCH],second_filter_material =  (concept_batches,list(zip(*sourceBatch))[C_IND_SOURCE_BATCH]))
                else:
                    t,p,tp = score_h.T_P_TP_Batch(rel_triples,list(zip(*sourceBatch))[R_IND_SOURCE_BATCH])
            for score_h in concept_scores:
                t,p,tp = score_h.T_P_TP_Batch(concept_batches,list(zip(*sourceBatch))[C_IND_SOURCE_BATCH])

        for i,data in enumerate(zip(idsBatch,sourceBatch,concept_pred_seq,concept_batches,rel_triples, graphs)):
            example_id, source,concept_pred,concept, rel_triple,graph= data
            if example_id in input_dict:
                input_snt = input_dict[example_id].input
            else:
                # input_snt = " ".join(source[TOK_IND_SOURCE_BATCH])
                input_snt = None
            mrp_graph, predicated_graph = decoder.graph_to_mrpGraph(example_id,
                                                                    graph,
                                                                    normalizeMod = opt.normalize_mod,
                                                                    flavor=frame2flavors_mapping[model.frame], framework=model.frame, sentence=input_snt)
            # we cannot use the token joints as sentence, which cause the anchors are very different
            # we should give adding the origin input sentence
            mrp_output.append(mrp_graph)
            out = []
            out.append( "# ::id "+ example_id +"\n")
            out.append( "# ::tok "+" ".join(source[TOK_IND_SOURCE_BATCH])+"\n")
            out.append(  "# ::lem "+" ".join(source[LEM_IND_SOURCE_BATCH])+"\n")
            out.append(  "# ::pos "+" ".join(source[POS_IND_SOURCE_BATCH])+"\n")
            out.append(  "# ::ner "+" ".join(source[NER_IND_SOURCE_BATCH])+"\n")
            out.append(  "# ::mwe "+" ".join(source[MWE_IND_SOURCE_BATCH])+"\n")
            out.append(  "# ::predicated "+" ".join([str(re_cat) for re_cat in concept_pred])+"\n")
            out.append(  "# ::transformed final predication "+" ".join([str(c) for c in concept])+"\n")
            out.append(  "# ::predicted rel"+" ".join([str(r) for r in rel_triple])+"\n")
            if set_name != "test":
                single_concept_scores = scorer.concept_score_initial(dicts)
                single_rel_scores = scorer.rel_scores_initial()
                gold_concept = source[C_IND_SOURCE_BATCH]
                gold_rel_triple = source[R_IND_SOURCE_BATCH]
                out.append(  "# ::gold_rel "+" ".join([str(r) for r in gold_rel_triple])+"\n")
                out.append(  "# ::gold_concept "+" ".join([str(c) for c in source[C_IND_SOURCE_BATCH]])+"\n")

                for score_h in single_rel_scores:
                    if score_h.second_filter:
                        t,p,tp = score_h.T_P_TP_Batch([rel_triple], [gold_rel_triple], second_filter_material =  ([concept],[gold_concept]))
                    else:
                        t,p,tp = score_h.T_P_TP_Batch([rel_triple],[gold_rel_triple])

                for score_h in single_concept_scores:
                    t,p,tp = score_h.T_P_TP_Batch([concept], [gold_concept])

                for score_h in single_concept_scores:
                    false_positive, false_negative = score_h.get_error(t = 0)
                    out.append(  "# ::"+ false_positive+"\n")
                    out.append(  "# ::"+ false_negative+"\n")

                for score_h in single_rel_scores:
                    false_positive, false_negative = score_h.get_error(t = 0)
                    out.append(  "# ::"+ false_positive+"\n")
                    out.append(  "# ::"+ false_negative+"\n")

            if decoder.frame == "amr":
                out.append( decoder.nodes_jamr(graph))
                out.append( decoder.edges_jamr(graph))
                out.append( predicated_graph)
            logger.info("".join(out)+"\n")
            batch_out[order[i]] = "".join(out)+"\n"
        output += batch_out
    total_out = "Smatch, P,R,F: "+ " ".join([str(i)for i in scorer.get_smatch(concept_scores, rel_scores)])
    logger.info(total_out)

    if set_name != "test":
        for score_h in concept_scores:
            score_h.show_error(t=0)

        for score_h in rel_scores:
            score_h.show_error(t=0)

        for score_h in concept_scores:
            logger.info(score_h)

        for score_h in rel_scores:
            logger.info(score_h)

    out_file = get_outfile(file, opt.result_folder, opt.output)
    with open(out_file, 'w+') as the_file:
        for data in output:
            the_file.write(data+'\n')
    logger.info("{} written.".format(out_file))

    out_mrp_file = get_mrp_outfile(file, opt.result_folder, opt.output)

    with open(out_mrp_file, 'w+') as the_file:
        for mrp_graph in mrp_output:
            if mrp_graph.input == None and mrp_graph.id in input_dict:
                mrp_graph.add_input(input_dict[mrp_graph.id].input)
            the_file.write(json.dumps(mrp_graph.encode(), indent=None, ensure_ascii = False))
            the_file.write("\n")
    logger.info("{} written.".format(out_mrp_file))
    return concept_scores,rel_scores,mrp_output


def create_mrp_dataiterators(frame, dicts):
    if frame == "amr":
        with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
        # take preprocessed pickle based file as input
        suffix = "." + frame+ "_pickle"+with_jamr+"_processed"
        rel_dict = dicts["amr_rel_dict"]
        input_snt_suffix = ".mrp_conllu"
    else:
        suffix = "."+ frame + "_pickle_processed"
        rel_dict = dicts["{}_rel_dict".format(frame)]
        input_snt_suffix = ".mrp_conllu"

    trainFolderPath = opt.build_folder+"/training/"
    trainingFilesPath = folder_to_files_path(trainFolderPath,suffix)

    devFolderPath = opt.build_folder +"/dev/"
    devFilesPath = folder_to_files_path(devFolderPath,suffix)

    testFolderPath = opt.build_folder+"/test/"
    testFilesPath = folder_to_files_path(testFolderPath,suffix)

    training_data_files = []
    dev_data_files = []
    test_data_files = []

    for file in testFilesPath:
        if opt.bert_model:
            test_data = BertDataIterator([file],opt,rel_dict)
        else:
            test_data = DataIterator([file],opt,rel_dict)
        test_data_files.append((test_data, file.replace(suffix, input_snt_suffix)))

    for file in devFilesPath:
        if opt.bert_model:
            dev_data = BertDataIterator([file],opt,rel_dict)
        else:
            dev_data = DataIterator([file],opt,rel_dict)
        dev_data_files.append((dev_data, file.replace(suffix, input_snt_suffix)))

    for file in trainingFilesPath:
        if opt.bert_model:
            training_data = BertDataIterator([file],opt,rel_dict)
        else:
            training_data = DataIterator([file],opt,rel_dict)
        training_data_files.append((training_data, file.replace(suffix, input_snt_suffix)))


    return {"test": test_data_files, "dev": dev_data_files, "training": training_data_files}

def create_mrp_decoder(frame, dicts):
    if frame == "amr":
        decoder = parser.AMRProcessors.AMRDecoder(opt,dicts)
    elif frame == "dm":
        decoder = parser.DMProcessors.DMDecoder(opt,dicts)
    elif frame == "psd":
        decoder = parser.PSDProcessors.PSDDecoder(opt,dicts)
    else:
        raise NotImplementedError("{} decoder is not supported".format(frame))

    decoder.eval()
    return decoder


def create_mrp_scorer(frame):
    if frame == "amr":
        return AMRNaiveScores()
    elif frame == "dm":
        return DMNaiveScores()
    elif frame == "psd":
        return PSDNaiveScores()
    else:
        raise NotImplementedError("{} is not supported".format(frame))


def main(opt):
    dicts = read_all_dicts(opt.build_folder, opt.frames)
    assert opt.restore_from

    decoder_dict = {}
    model_dict = {}
    data_iterator_dict = {}
    component_dict = {}
    scorer_dict = {}

    # TODO: to support all
    for frame in opt.frames:
        decoder_dict[frame] = create_mrp_decoder(frame, dicts)
        data_iterator_dict[frame] = create_mrp_dataiterators(frame, dicts)
        model, parameters,optt, optim =  load_old_model(frame,dicts,opt,True)
        model_dict[frame] = model
        scorer_dict[frame] = create_mrp_scorer(frame)
        opt.start_epoch =  1
        logger.info(opt.restore_from+"\n")
        logger.info(":\n{}".format(model))
        logger.info("optt:{}".format(optt))
        logger.info("opt:{}".format(opt))
        logger.info('processing testing {}'.format(frame))
        for test_data, file in data_iterator_dict[frame]["test"]:
            concept_scores,rel_scores,mrp_graphs=generate_graph("test", model_dict[frame],decoder_dict[frame],scorer_dict[frame],test_data,dicts,file)

        logger.info('processing validation {}'.format(frame))
        for dev_data, file in data_iterator_dict[frame]["dev"]:
            concept_scores,rel_scores,mrp_graphs=generate_graph("dev",model_dict[frame],decoder_dict[frame],scorer_dict[frame], dev_data,dicts,file)

        logger.info('processing training{}'.format(frame))
        for training_data, file in data_iterator_dict[frame]["training"]:
            concept_scores,rel_scores,mrp_graphs=generate_graph("training",model_dict[frame],decoder_dict[frame],scorer_dict[frame], training_data,dicts,file)

if __name__ == "__main__":
    global opt
    opt = generate_parser().parse_args()
    opt.lemma_dim = opt.dim
    opt.high_dim = opt.dim

    opt.cuda = len(opt.gpus)

    logger.info("opt:{}".format(opt))

    if torch.cuda.is_available() and not opt.cuda:
        logger.warn("WARNING: You have a CUDA device, so you should probably run with -cuda")

    ## it is used to specify the gpus to use
    ## it is not recommended for using this.
    #if opt.cuda and opt.gpus[0] != -1:
    #    cuda.set_device(opt.gpus[0])
    main(opt)
