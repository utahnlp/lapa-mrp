#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts build dictionary and data into numbers, and seralize into tree file.

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from utility.ucca_utils.UCCAStringCopyRules import *
from utility.ucca_utils.UCCAReCategorization import *
from utility.data_helper import *
from utility.constants import *
from parser.Dict import *
from parser.UCCAProcessors import *
import logging
from parser.modules.bert_utils import MRPBertTokenizer
from parser.modules.char_utils import CharTokenizerUtils

import argparse

logger = logging.getLogger("mrp")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def ucca_data_build_parser():
    parser = argparse.ArgumentParser(description='ucca_data_build.py')

    ## Data options
    parser.add_argument('--threshold', default=10, type=int,
                        help="""threshold for high frequency concepts""")

    parser.add_argument('--skip', default=0, type=int,
                        help="""skip dict build if dictionary already built""")
    parser.add_argument('--merge_common_dicts', default=1, type=int,
                        help="""whether to merge common dict if already existed""")
    parser.add_argument('--suffix', default=".mrp_ucca", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--companion_suffix', default=".mrp_conllu_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--build_folder', default="", type=str,
                        help="""the build folder for dict and rules, data""")
    # https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/pytorch_transformers/tokenization_bert.py
    parser.add_argument('--bert_model', default="bert-base-cased", type=str,
                        help="""the name for pretraind bert name""")
    parser.add_argument('--do_lower_case', default=False, type=bool,
                        help="""Whether to lower case when tokenization""")
    return parser


parser = ucca_data_build_parser()
opt = parser.parse_args()
# here use bert to preprare token ids, used for fine tuning.
# if for static bert, we can do that offline, save an encoding for each utterance
bert_tokenizer = MRPBertTokenizer.from_pretrained(opt.bert_model, do_lower_case=opt.do_lower_case)

suffix = opt.suffix
trainFolderPath = opt.build_folder + "/training/"
trainingTreeFile = opt.build_folder + "/training/training.ucca_tree_processed"
trainingFilesPath = folder_to_files_path(trainFolderPath, suffix)
trainingCompanionFilesPath = folder_to_files_path(trainFolderPath, opt.companion_suffix)

devFolderPath = opt.build_folder + "/dev/"
devTreeFile = opt.build_folder + "/dev/dev.ucca_tree_processed"
devFilesPath = folder_to_files_path(devFolderPath, suffix)
devCompanionFilesPath = folder_to_files_path(devFolderPath, opt.companion_suffix)

testFolderPath = opt.build_folder + "/test/"
testTreeFile = opt.build_folder + "/test/test.ucca_tree_processed"
testFilesPath = folder_to_files_path(testFolderPath, suffix)
testCompanionFilesPath = folder_to_files_path(testFolderPath, opt.companion_suffix)

def generate_tree(data):
    input_snt = data["input_snt"]
    snt_token = data["tok"]
    tok_anchors = data["anchors"]

    if "ucca_t" in data:
        ucca_t = data["ucca_t"]
    else:
        ucca_t = None

    if 'mrp_ucca' in data:
        ucca = UCCAGraph(ucca_t, data['mrp_ucca'], snt_token, tok_anchors)
        return ucca.tree
    else:
        # TODO, for other framworks
        return None

def buildTreeData(all_data, outputPTBFile=False):
    n = 0
    # now only handle the ucca data, to extend for all other formats
    ucca_tree_data = []
    with open(outputPTBFile, "w+") as ofile:
        for id, data in all_data.items():
            n = n + 1
            if n %1000 == 0:
                logger.info(n)
            # try to extend this to other frameworks
            if 'mrp_ucca' in data:
                ptb_tree = generate_tree(data)
                ofile.write(ptb_tree.linearize())
                ofile.write("\n")
                ucca_tree_data.append(ptb_tree)

    logger.info("{} has been preprocessed, {} are good for training".format(n, len(ucca_tree_data)))
    return len(ucca_tree_data)

# after dict has been saved, load them and make the whole dataset into tree based data point
logger.info("processing training set")
training_data = readFeaturesInput(trainingCompanionFilesPath)
mergeWithAnnotatedGraphs(training_data,trainingFilesPath)
buildTreeData(training_data, trainingTreeFile)

logger.info(("processing development set"))
dev_data = readFeaturesInput(devCompanionFilesPath)
mergeWithAnnotatedGraphs(dev_data, devFilesPath)
buildTreeData(dev_data, devTreeFile)

logger.info("processing test set")
test_data = readFeaturesInput(testCompanionFilesPath)
#mergeWithAnnotatedGraphs(test_data, testFilesPath)
buildTreeData(test_data, testTreeFile)
