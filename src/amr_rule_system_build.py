#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts to build AMRStringCopyRules and AMRReCategorizor

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.
@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from utility.amr_utils.AMRStringCopyRules import *
from utility.amr_utils.AMRReCategorization import *
from utility.data_helper import *
import logging

import argparse

logger = logging.getLogger("mrp.amr_rule_system_build")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def arg_parser():
    parser = argparse.ArgumentParser(description='amr_rule_system_build.py')

    ## Data options
    parser.add_argument('--threshold', default=5, type=int,
                        help="""threshold for non-aligned high frequency concepts""")

    parser.add_argument('--jamr', default=0, type=int,
                        help="""wheather to enhance string matching with additional jamr alignment""")
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
suffix = opt.suffix + "_jamr" if opt.jamr else opt.suffix
with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
trainFolderPath = opt.build_folder+"/training/"
trainingFilesPath = folder_to_files_path(trainFolderPath,suffix)
trainingCompanionFilesPath = folder_to_files_path(trainFolderPath, opt.companion_suffix)

lock = threading.Lock()
def add_count(store,new,additional=None):
    lock.acquire()

    for i in new:
        # for non_set_set, i is an unaligned AMRUniversal
        # value is 2 element array, 0 is the count, 1 is additional (snt_str)
        if not i in store:
            store[i] = [1,[additional]]
        else:
            store[i][0] = store[i][0] + 1
            store[i][1].append(additional)
    lock.release()

def handle_sentence(data,n,update_freq,use_template,jamr = False):
    if n % 500 == 0:
        logger.info(n)

    snt_token = data["tok"]
    pos_token = data["pos"]
    lemma_token = data["lem"]
    if "amr_t" in data:
        amr_t = data["amr_t"]
    else:
        amr_t = None

    if "aligns" in data:
        aligns = data["align"]
    else:
        aligns = ""

    amr = AMRGraph(amr_t, data['mrp_amr'], aligns=aligns)
    lemma_str  =" ".join(lemma_token)

    if use_template:
        fragment_to_node_converter.match(amr,rl ,snt_token,lemma_token,pos_token,lemma_str,jamr=jamr )

    fragment_to_node_converter.convert(amr,rl ,snt_token,lemma_token,pos_token,lemma_str )
    results = rl.get_matched_concepts(snt_token,amr,lemma_token,pos_token,with_target=update_freq,jamr=jamr)
    # after aligning, (if jamr existed, mege from jamr), count the freqeuncy again
    # for other MR, here we only need to do the counting once.
    if update_freq:
        for n_c_a in results:
            # n is node, c is it value AMEUniversal, a is align, an array of tuples
            # align =[(a, lemma[a], pos[a])], here a is a token index
            for i_le in n_c_a[2]:
                # i_le = (a, lemma, pos)
                # i_le[1] is lemma of that token
                # n_c_a[1] is the value AMRUniversal
                rl.add_lemma_freq(i_le[1],n_c_a[1].le,n_c_a[1].cat,sense = n_c_a[1].sense)

    snt_str = " ".join(snt_token)
    # none_rule is an array of AMRUniversal, those are not aligned.
    none_rule = [n_c_a[1] for n_c_a in results if len(n_c_a[2])==0]
    add_count(non_rule_set,none_rule,snt_str)

def readFile(all_data, update_freq=False,use_template=True):
    n = 0
    for id, data in all_data.items():
        if 'mrp_amr' in data:
            n=n+1
            if opt.jamr:
                handle_sentence(data,n,update_freq,use_template,jamr=True)
            else:
                handle_sentence(data,n,update_freq,use_template,jamr=False)
    return n

rl = AMRRules()
non_rule_set = dict()
logger.info(("reading "+','.join(trainingCompanionFilesPath)+"......"))
training_data = readFeaturesInput(trainingCompanionFilesPath)
logger.info("read done "+ str(len(training_data)) +" setences")
num_merged_graphs = mergeWithAnnotatedGraphs(training_data, trainingFilesPath)
logger.info("merging with graphs: " + str(num_merged_graphs))
# the Recategorize to categorize the graph into a fake node
fragment_to_node_converter = AMRReCategorizor(path=opt.build_folder+"/dicts/graph_to_node_dict", training=True)
#
non_rule_set_last = non_rule_set
# initial the lemma_cheat, with just the propbank, verbalization list
rl.build_lemma_cheat()
#
non_rule_set = dict()

# add reading sentence and extract rules.
n = readFile(training_data,update_freq=True,use_template=True)
logger.info(("done reading "+ '.'.join(trainingFilesPath)+", "+str(n)+" sentences processed"))
#non_rule_set = non_rule_set_last
# after using templates, and other rules, and string matching, recompute the non_rule_set
high_text_num,high_frequency,low_frequency,low_text_num=unmixe(non_rule_set,threshold )
logger.info("initial converted,threshold={0},len(non_rule_set)={1},high_text_num={2},high_frequency={3},low_frequency={4},low_text_num={5}".format(
    threshold,
    len(non_rule_set),
    len(high_text_num),
    len(high_frequency),
    len(low_frequency),
    len(low_text_num)))
#logger.info("len(concept_embedding)".format(len(concept_embedding)))
#
#
#
non_rule_set_initial_converted = non_rule_set
# after using template, string distance to align, update the lemma_cheat dict
rl.build_lemma_cheat()
fragment_to_node_converter.save(path=opt.build_folder+"/dicts/graph_to_node_dict_extended"+with_jamr)
fragment_to_node_converter = AMRReCategorizor(from_file=False, path=opt.build_folder+"dicts/graph_to_node_dict_extended"+with_jamr,training=False)
rl.save(opt.build_folder+"dicts/amr_rule_f"+with_jamr)
non_rule_set = dict()
NERS = {}

#need to rebuild copying dictionary again based on recategorized graph

n = readFile(training_data,update_freq=False,use_template=False)
logger.info(("done reading "+ ','.join(trainingFilesPath) +", "+str(n)+" sentences processed"))

non_rule_set_f = Pickle_Helper(opt.build_folder+"dicts/amr_non_rule_set")
non_rule_set_f.dump(non_rule_set,"amr_non_rule_set")
non_rule_set_f.save()

#only intermediate data, won't be useful for final parser
non_rule_set_f = Pickle_Helper(opt.build_folder+"/dicts/amr_non_rule_set")
non_rule_set_f.dump(non_rule_set_last,"initial_amr_non_rule_set")
non_rule_set_f.dump(non_rule_set_initial_converted,"initial_converted_amr_non_rule_set")
non_rule_set_f.dump(non_rule_set,"amr_non_rule_set")
non_rule_set_f.save()


high_text_num,high_frequency,low_frequency,low_text_num=unmixe(non_rule_set,threshold )
logger.info("final converted,threshold={0},len(non_rule_set)={1},high_text_num={2},high_frequency={3},low_frequency={4},low_text_num={5}".format(
    threshold,
    len(non_rule_set),
    len(high_text_num),
    len(high_frequency),
    len(low_frequency),
    len(low_text_num)))
