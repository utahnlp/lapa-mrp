#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to build EDSStringCopyRules and EDSReCategorizor

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.
@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from utility.eds_utils.EDSStringCopyRules import *
from utility.eds_utils.EDSReCategorization import *
from utility.data_helper import *
import logging

import argparse

logger = logging.getLogger("mrp.eds_rule_system_build")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def arg_parser():
    parser = argparse.ArgumentParser(description='eds_rule_system_build.py')

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
        # for non_set_set, i is an unaligned EDSUniversal
        # value is 2 element array, 0 is the count, 1 is additional (snt_str)
        if not i in store:
            store[i] = [1,[additional]]
        else:
            store[i][0] = store[i][0] + 1
            store[i][1].append(additional)
    lock.release()

def add_count_kv(store,new):
    lock.acquire()

    for i,additional in new:
        # for non_set_set, i is an unaligned EDSUniversal
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
    mwe_token = data["mwe"]
    lemma_token = data["lem"]
    if "eds_t" in data:
        eds_t = data["eds_t"]
    else:
        eds_t = None

    # no alignments required for eds
    eds = EDSGraph(eds_t, data['mrp_eds'])
    lemma_str  = " ".join(lemma_token)

    # TODO: For EDS, it is good to combine two share node into one. especially for mwe, and compund ners
    # match character offset with token before converting
    results = rl.get_matched_concepts(data["input_snt"], snt_token,eds,lemma_token,pos_token,mwe_token, tok_anchors)
    # we already have the alignment for each node.
    if update_freq:
        # this result only contain the transformed nodes
        for n, c, a, flags in results:
            # n is node, c is it value AMEUniversal, a is align, an array of tuples
            for i in a:
                # use whole frame for dictionary, use the predicate le, pos, sense
                rl.add_lemma_pos_freq(snt_token[i], lemma_token[i], pos_token[i], c)
        # fragment_to_node_converter.read_senses(eds)

    # after aligned to the token, we normalize the copy menachnism
    # COPY_WORD, COPY_LEMMA, if not either of them, adding into none_rule set
    snt_str = " ".join(snt_token)
    none_rule = [c.no_anchor_copy() for n,c,a,copy in results if not copy]
    none_align = [(c.le,data["input_snt"][c.anchors[0]["from"]:c.anchors[0]["to"]]) for n,c,a,copy in results if len(a) == 0]
    add_count_kv(unaligned_set, none_align)
    add_count(non_rule_set, none_rule, snt_str)


def readFile(all_data, update_freq=False, use_combine=False):
    n = 0
    for id, data in all_data.items():
        if 'mrp_eds' in data:
            n=n+1
            data = rl.annotate_mwe(data)
            handle_sentence(data,n,update_freq, use_combine)
    return n

rl=EDSRules()
non_rule_set = dict()
unaligned_set = dict()
fragment_to_node_converter= EDSReCategorizor(from_file=False)
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
high_frequency,low_frequency=EDSRules.unmixe(non_rule_set,threshold)
logger.info("initial converted,threshold={},len(non_rule_set)={},high_frequency={},low_frequency={}".format(
    threshold,
    len(non_rule_set),
    len(high_frequency),
    len(low_frequency)))
#logger.info("{}".format(str(high_frequency)))

high_unaligned,low_unaligned =EDSRules.unmixe(unaligned_set,0)
logger.info("initial converted,threshold={},len(unaligned_set)={},high_unaligned={},low_unaligned={}".format(
    threshold,
    len(non_rule_set),
    len(high_unaligned),
    len(low_unaligned)))

for x in high_unaligned:
    logger.info("{} -> {}".format(x, high_unaligned[x]))



rl.build_lemma_cheat()
fragment_to_node_converter.save(path=opt.build_folder+"/dicts/eds_recategorization")
fragment_to_node_converter = EDSReCategorizor(from_file=False, path=opt.build_folder+"dicts/eds_recategorization",training=False)
rl.save(opt.build_folder+"dicts/eds_rule_f")

non_rule_set_f = Pickle_Helper(opt.build_folder+"dicts/eds_non_rule_set")
non_rule_set_f.dump(non_rule_set,"eds_non_rule_set")
non_rule_set_f.save()
