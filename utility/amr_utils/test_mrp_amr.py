#!/usr/bin/env python3.6
# coding=utf-8
'''
Load amr transformed mrp, and load into AMRGraph as gold label, test the output with the gold AMRGraph
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-06-23
'''

import parser
import os
import sys
from parser.AMRProcessors import *
from utility.data_helper import folder_to_files_path
from src import *
from parser.Dict import read_dicts
from src.config_reader import get_parser
from utility.mtool.score import mces
from utility.constants import *

logger = logging.getLogger("amr.generator")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def arg_parser():
    parser = argparse.ArgumentParser(description='test_mrp_amr.py')

    ## Data options
    parser.add_argument('--jamr', default=0, type=int,
                        help="""wheather to add .jamr at the end""")
    parser.add_argument('--suffix', default=".mrp_amr", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--companion_suffix', default=".mrp_conllu_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--build_folder', default="", type=str,
                        help="""the build folder for dict and rules, data""")
    return parser

def test_amr_pipe(data, n):
    if "amr_t" in data:
        amr_t = data["amr_t"]
    else:
        amr_t = None

    if "mrp_amr" in data:
        mrp_amr = data["mrp_amr"]
    else:
        mrp_amr = None

    # gold amrGraph
    amr = AMRGraph(amr_t, mrp_amr)
    graph = amr.graph
    h_v = BOS_WORD
    root_v = amr.root
    root_symbol = AMRUniversal(BOS_WORD,BOS_WORD,NULL_WORD)
    graph.add_node(h_v, value=root_symbol, align=-1,gold=True,dep=1)
    graph.add_edge(h_v, root_v, key=":top", role=":top")
    graph.add_edge(root_v, h_v, key=":top-of", role=":top-of")
    # to mrp
    mrp_amr_back, amr_string = AMRDecoder.graph_to_mrpGraph(mrp_amr.id, graph, normalizeMod=True, flavor=2, framework="amr")
    mrp_amr_back.add_input(mrp_amr.input)
    mrp_amr_back.normalize(actions=['edges','case','attributes'])
    mrp_amr.normalize(actions=['edges','case','attributes'])
    result = mces.evaluate([mrp_amr], [mrp_amr_back], trace = 2);
    #result = mrp_smatch.evaluate([mrp_amr], [mrp_amr_back], format="json", limit= 10, trace=2)
    if result["all"]["f"] != 1.0:
        logger.error("result is {} for {}\n grpah_nodes:{}\n  graph_edges:{}\n gold:{}\n pred:{}\n amr:{} \n\n".format(result,mrp_amr.id, graph.nodes.data(), graph.edges.data(), mrp_amr.encode(), mrp_amr_back.encode(), amr_string))

def test_mrp_dataset(dataset):
    n = 0
    for id, data in dataset.items():
        # try to extend this to other frameworks
        if 'mrp_amr' in data:
            n = n + 1
            if n % 500 ==0:
                logger.info(n)

            test_amr_pipe(data, n)

def test_mrp_to_graph():
    logger.info("processing training set")
    training_data = readFeaturesInput(opt.trainingCompanionFilesPath)
    mergeWithAnnotatedGraphs(training_data, opt.trainingFilesPath)
    test_mrp_dataset(training_data)

    logger.info(("processing development set"))
    dev_data = readFeaturesInput(opt.devCompanionFilesPath)
    mergeWithAnnotatedGraphs(dev_data, opt.devFilesPath)
    test_mrp_dataset(dev_data)

    logger.info("processing test set")
    test_data = readFeaturesInput(opt.testCompanionFilesPath)
    mergeWithAnnotatedGraphs(test_data, opt.testFilesPath)
    test_mrp_dataset(test_data)


def main(opt):
    test_mrp_to_graph()

if __name__ == "__main__":
    global opt
    parser = arg_parser()
    opt = parser.parse_args()
    suffix = opt.suffix + "_jamr" if opt.jamr else opt.suffix
    with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
    opt.trainFolderPath = opt.build_folder + "/training/"
    opt.trainingFilesPath = folder_to_files_path(opt.trainFolderPath, suffix)
    opt.trainingCompanionFilesPath = folder_to_files_path(opt.trainFolderPath, opt.companion_suffix)

    opt.devFolderPath = opt.build_folder + "/dev/"
    opt.devFilesPath = folder_to_files_path(opt.devFolderPath, suffix)
    opt.devCompanionFilesPath = folder_to_files_path(opt.devFolderPath, opt.companion_suffix)

    opt.testFolderPath = opt.build_folder + "/test/"
    opt.testFilesPath = folder_to_files_path(opt.testFolderPath, suffix)
    opt.testCompanionFilesPath = folder_to_files_path(opt.testFolderPath, opt.companion_suffix)

    logger.info("opt:{}".format(opt))

    main(opt)
