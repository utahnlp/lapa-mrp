#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to build DMStringCopyRules and DMReCategorizor

Data path information should also be specified here for
trainFolderPath, devFolderPath
as we allow option to choose from two version of data.
@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from utility.dm_utils.DMStringCopyRules import *
from utility.dm_utils.DMReCategorization import *
from utility.data_helper import *
from utility.constants import *
from parser.DMProcessors import *
import logging

import argparse

logger = logging.getLogger("mrp.dm_rule_system_build")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def arg_parser():
    parser = argparse.ArgumentParser(description='dm_rule_system_build.py')

    ## Data options
    parser.add_argument('--threshold', default=5, type=int,
                        help="""threshold for non-aligned high frequency concepts""")
    parser.add_argument('--suffix', default=".mrp", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--companion_suffix', default=".mrp_conllu_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--build_folder', default="", type=str ,
                        help="""the folder""")
    return parser

parser = arg_parser()
opt = parser.parse_args()
threshold = opt.threshold
suffix = opt.suffix
trainFolderPath = opt.build_folder+"/training/"
trainingFilesPath = folder_to_files_path(trainFolderPath,suffix)
trainingCompanionFilesPath = folder_to_files_path(trainFolderPath, opt.companion_suffix)

lock = threading.Lock()
def add_count(store,new,additional=None):
    lock.acquire()

    for i in new:
        # for non_set_set, i is an unaligned DMUniversal
        # value is 2 element array, 0 is the count, 1 is additional (snt_str)
        if not i in store:
            store[i] = [1,[additional]]
        else:
            store[i][0] = store[i][0] + 1
            store[i][1].append(additional)
    lock.release()

def handle_sentence(data,n,update_freq, use_combine):
    if n % 500 == 0:
        logger.info(n)

    # str
    input_snt = data["input_snt"]
    snt_token = data["tok"]
    tok_anchors = data["anchors"]
    pos_token = data["pos"]
    ner_token = data["ner"]
    mwe_token = data["mwe"]
    lemma_token = data["lem"]
    if "dm_t" in data:
        dm_t = data["dm_t"]
    else:
        dm_t = None

    # no alignments required for dm
    dm = DMGraph(dm_t, data['mrp_dm'])
    lemma_str  = " ".join(lemma_token)

    # TODO: For DM, it is good to combine two share node into one. especially for mwe, and compund ners
    # match character offset with token before converting
    rl.get_matched_concepts(snt_token,dm,lemma_token,pos_token, tok_anchors, ner_token, mwe_token)
    if use_combine:
        # when converting, node meerged, anchors also merged.
        fragment_to_node_converter.convert(dm, rl ,snt_token,lemma_token,pos_token,lemma_str)

    results = rl.get_matched_concepts(snt_token,dm,lemma_token,pos_token, tok_anchors, ner_token, mwe_token)
    # we already have the alignment for each node.
    if update_freq:
        for n, c, a, can_copy, can_le_copy in results:
            # n is node, c is it value AMEUniversal, a is align, an array of tuples
            for i in a:
                # use whole frame for dictionary, use the predicate le, pos, sense
                # TODO, this may not be correct for  MWE
                if ner_token[i] == 'O':
                    rl.add_lemma_pos_freq(snt_token[i], lemma_token[i], c)
        fragment_to_node_converter.read_senses(dm)

    # after aligned to the token, we normalize the copy menachnism
    # COPY_WORD, COPY_LEMMA, if not either of them, adding into none_rule set
    snt_str = " ".join(snt_token)
    none_rule = [c.no_anchor_copy() for n,c,a,copy,can_le_copy in results if not copy]
    add_count(non_rule_set, none_rule, snt_str)
    none_le_rule = [c.no_anchor_copy() for n,c,a,copy,can_le_copy in results if not can_le_copy]
    add_count(non_le_rule_set, none_le_rule, snt_str)


def readFile(all_data, update_freq=False, use_combine=False):
    n = 0
    for id, data in all_data.items():
        if 'mrp_dm' in data:
            n=n+1
            data = input_preprocessor.annotate_mwe(data)
            handle_sentence(data,n,update_freq, use_combine)
    return n

rl=DMRules()
input_preprocessor = DMInputPreprocessor(opt, core_nlp_url, dm_mwe_file)
non_rule_set = dict()
non_le_rule_set = dict()
fragment_to_node_converter= DMReCategorizor(from_file=False)
logger.info(("reading "+','.join(trainingCompanionFilesPath)+"......"))
training_data = readFeaturesInput(trainingCompanionFilesPath)
logger.info("read done "+ str(len(training_data)) +" setences")
num_merged_graphs = mergeWithAnnotatedGraphs(training_data, trainingFilesPath)
logger.info("merging with graphs: " + str(num_merged_graphs))

# build lemma only be surface smi frames
rl.build_lemma_cheat()

# add reading sentence and extract rules
# TODO: now, didn't combine more mwe frames, ner frames
n = readFile(training_data,update_freq=True, use_combine=False)
# n = readFile(training_data,update_freq=True, use_combine=True)
logger.info(("done reading "+ '.'.join(trainingFilesPath)+", "+str(n)+" sentences processed"))


# TODO: to
#high_frequency,low_frequency=PSDRules().unmixe(non_rule_set,threshold)
high_frequency,low_frequency=DMRules.unmixe(non_rule_set,threshold)
logger.info("initial converted,threshold={},len(non_rule_set)={}, len(non_le_rule_set)={},high_frequency={},low_frequency={}".format(
    threshold,
    len(non_rule_set),
    len(non_le_rule_set),
    len(high_frequency),
    len(low_frequency)))
logger.info("{}".format(str(high_frequency)))


rl.build_lemma_cheat()
fragment_to_node_converter.save(path=opt.build_folder+"/dicts/dm_recategorization")
fragment_to_node_converter = DMReCategorizor(from_file=False, path=opt.build_folder+"dicts/dm_recategorization",training=False)
rl.save(opt.build_folder+"dicts/dm_rule_f")

non_rule_set_f = Pickle_Helper(opt.build_folder+"dicts/dm_non_rule_set")
non_rule_set_f.dump(non_rule_set,"dm_non_rule_set")
non_rule_set_f.save()

non_le_rule_set_f = Pickle_Helper(opt.build_folder+"dicts/dm_non_le_rule_set")
non_le_rule_set_f.dump(non_le_rule_set,"dm_non_le_rule_set")
non_le_rule_set_f.save()
