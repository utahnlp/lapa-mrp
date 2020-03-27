#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to build PSDStringCopyRules and PSDReCategorizor

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.
@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-05-30
'''

from utility.psd_utils.PSDStringCopyRules import *
from utility.psd_utils.PSDReCategorization import *
from utility.data_helper import *
from parser.PSDProcessors import *
from utility.constants import *
import logging

import argparse

logger = logging.getLogger("mrp.psd_rule_system_build")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def arg_parser():
    parser = argparse.ArgumentParser(description='psd_rule_system_build.py')

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
        # for non_set_set, i is an unaligned PSDUniversal, we cannot use the anochors there.
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

    snt_token = data["tok"]
    tok_anchors = data["anchors"]
    pos_token = data["pos"]
    lemma_token = data["lem"]
    mwe_token = data["mwe"]
    if "psd_t" in data:
        psd_t = data["psd_t"]
    else:
        psd_t = None

    # no alignments required for psd
    psd = PSDGraph(psd_t, data['mrp_psd'])
    lemma_str  = " ".join(lemma_token)

    # TODO: For PSD, it is good to combine two share node into one. especially for mwe, and compund ners
    # match character offset with token before converting
    rl.get_matched_concepts(snt_token,psd,lemma_token,pos_token, mwe_token, tok_anchors)
    if use_combine:
        # when converting, node meerged, anchors also merged.
        fragment_to_node_converter.convert(psd, rl ,snt_token,lemma_token,pos_token,lemma_str, mwe_token)
    results = rl.get_matched_concepts(snt_token,psd,lemma_token,pos_token, mwe_token, tok_anchors)
    # we already have the alignment for each node.
    if update_freq:
        for n, c, a, flags in results:
            # n is node, c is it value Universal, a is align, an array of tuples
            for i in a:
                # i here is the index of the token, when copy we use the le first, then fixed it with some lemmatize cheat
                rl.add_lemma_freq(lemma_token[i], c.le, sense = c.sense)
        fragment_to_node_converter.read_senses(psd)

    # after aligned to the token, we normalize the copy menachnism
    # COPY_WORD, COPY_LEMMA, if not either of them, adding into none_rule set
    snt_str = " ".join(snt_token)
    none_rule = [c.no_anchor_copy() for n,c,a,copy in results if not copy]
    add_count(non_rule_set, none_rule, snt_str)


def readFile(all_data, update_freq=False, use_combine=False):
    n = 0
    for id, data in all_data.items():
        if 'mrp_psd' in data:
            n=n+1
            data = input_preprocessor.annotate_mwe(data)
            handle_sentence(data,n,update_freq, use_combine)
    return n

rl=PSDRules()
input_preprocessor = PSDInputPreprocessor(opt, core_nlp_url)
non_rule_set = dict()
fragment_to_node_converter= PSDReCategorizor(from_file=False, training=False)
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
high_frequency,low_frequency=PSDRules.unmixe(non_rule_set,threshold)
logger.info("initial converted,threshold={},len(non_rule_set)={},high_frequency={},low_frequency={}".format(
    threshold,
    len(non_rule_set),
    len(high_frequency),
    len(low_frequency)))
logger.info("{}".format(str(high_frequency)))

rl.build_lemma_cheat()
fragment_to_node_converter.save(path=opt.build_folder+"/dicts/psd_recategorization")

fragment_to_node_converter = PSDReCategorizor(from_file=False, path=opt.build_folder+"dicts/psd_recategorization",training=False)
rl.save(opt.build_folder+"dicts/psd_rule_f")

non_rule_set_f = Pickle_Helper(opt.build_folder+"dicts/psd_non_rule_set")
non_rule_set_f.dump(non_rule_set,"psd_non_rule_set")
non_rule_set_f.save()
