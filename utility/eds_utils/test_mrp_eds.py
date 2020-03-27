#!/usr/bin/env python3.6
# coding=utf-8
'''

Load eds transformed mrp, and load into AMRGraph as gold label, test the output with the gold AMRGraph
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-06-23
'''

import parser
import os
import sys
from torch import cuda
from parser.EDSProcessors import *
from utility.eds_utils.EDSGraph import *
from utility.data_helper import folder_to_files_path
from src import *
from parser.Dict import read_dicts
from src.config_reader import get_parser
from utility.mtool.score import mces
from utility.constants import *

logger = logging.getLogger("mrp.eds.generator")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def arg_parser():
    parser = argparse.ArgumentParser(description='test_mrp_eds.py')

    ## Data options
    parser.add_argument('--suffix', default=".mrp_eds", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--companion_suffix', default=".mrp_conllu_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--build_folder', default="", type=str,
                        help="""the build folder for dict and rules, data""")
    return parser

def test_eds_pipe(data, n):
    if "eds_matrix" in data:
        eds_t = data["eds_matrix"]
    else:
        eds_t = None

    if "mrp_eds" in data:
        mrp_eds = data["mrp_eds"]
    else:
        mrp_eds = None

    # gold edsGraph
    eds = EDSGraph(eds_t, mrp_eds)
    graph = eds.graph
    h_v = BOS_WORD
    root_v = eds.root
    root_symbol = EDSUniversal.TOP_EDSUniversal()
    graph.add_node(h_v, value=root_symbol, align=[],gold=True,dep=1)
    graph.add_edge(h_v, root_v, key=":top", role=":top")
    graph.add_edge(root_v, h_v, key=":top-of", role=":top-of")
    # to mrp
    mrp_eds_back, _ = EDSDecoder.graph_to_mrpGraph(mrp_eds.id, graph, flavor=1, framework="eds", sentence = mrp_eds.input)
    result = mces.evaluate([mrp_eds], [mrp_eds_back], trace = 2);
    if result["all"]["f"] != 1.0:
        logger.error("result is {} for {}\n grpah_nodes:{}\n  graph_edges:{}\n gold:{}\n pred:{} \n\n".format(result,mrp_eds.id, graph.nodes.data(), graph.edges.data(), mrp_eds.encode(), mrp_eds_back.encode()))

def test_mrp_dataset(dataset):
    n = 0
    for id, data in dataset.items():
        # try to extend this to other frameworks
        if 'mrp_eds' in data:
            n = n + 1
            if n % 500 ==0:
                logger.info(n)

            test_eds_pipe(data, n)

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
    suffix = opt.suffix
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
