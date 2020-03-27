#!/usr/bin/env python3.6
# coding=utf-8
'''

Load ucca transformed mrp, and load into UCCA trees as gold label, test the output with the gold UCCA MRP graph
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-06-23
'''

import os
import traceback
import sys
from utility.ucca_utils.UCCAGraph import *
from parser.UCCAProcessors import UCCADecoder
from utility.ptb_utils.ptb2ucca import *
from utility.data_helper import folder_to_files_path
from src import *
from parser.Dict import read_dicts
from src.config_reader import get_parser
from utility.mtool.score import mces
from utility.constants import *

logger = logging.getLogger("mrp.ucca.generator")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def arg_parser():
    parser = argparse.ArgumentParser(description='test_mrp_ucca.py')

    ## Data options
    parser.add_argument('--suffix', default=".mrp_ucca", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--companion_suffix', default=".mrp_conllu_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--build_folder', default="", type=str,
                        help="""the build folder for dict and rules, data""")
    return parser

def test_ucca_fixed_pipe(mrp_ucca, ucca, snt_token, tok_anchors):
    # need a step transform tree to graph by adding missing reentrance edges

    # fixed_graph hixed discontinuous, but with reentracne
    new_graph = ucca.fixed_graph

    for (h_role_align, d_role_align, ori_h_role_align) in ucca.fixed:
        d_role = d_role_align[0][1]
        ori_h_node = find_node_by_role_aligns(new_graph, ori_h_role_align)
        h_node = find_node_by_role_aligns(new_graph, h_role_align)
        d_node = find_node_by_role_aligns(new_graph, d_role_align)
        try:
            if ori_h_node!=None and h_node!=None and d_node!=None:
                new_graph.remove_edge(ori_h_node, d_node)
                new_graph.remove_edge(d_node, ori_h_node)
                new_graph.add_edge(h_node, d_node, key=d_role, role=d_role)
                new_graph.add_edge(d_node, h_node, key=d_role+"-of", role=d_role+"-of")
            else:
                logger.error("missing some discontinuous:{},{}".format((h_role_align, d_role_align, ori_h_role_align), (h_node, d_node, ori_h_node)))
        except:
            logger.error(traceback.format_exc())
            logger.error("error happend when fix discontinuous nodes in {}, {}".format((h_role_align, d_role_align, ori_h_role_align), (ori_h_node, h_node, d_node)))

    #logger.error("new_graph after adding reentrance and fixed: for {}\n graph_nodes:{}\n  graph_edges:{}\n\n".format(mrp_ucca.id, new_graph.nodes.data(), new_graph.edges.data()))
    mrp_ucca_fixed_back, M = UCCADecoder.graph_to_mrpGraph(mrp_ucca.id, new_graph, flavor=2, framework="ucca")
    mrp_ucca_fixed_back.add_input(mrp_ucca.input)
    new_result = mces.evaluate([mrp_ucca], [mrp_ucca_fixed_back], trace = 2);
    if new_result["all"]["f"] != 1.0:
        # logger.error("graph_nodes:{}\n  graph_edges:{}\n".format(graph.nodes.data(), graph.edges.data()))
        logger.error("fixed_graph: result is {} for {}\n graph_nodes:{}\n  graph_edges:{}\n gold:{}\n pred:{} \n\n".format(new_result, mrp_ucca.id, M.nodes.data(), M.edges.data(), mrp_ucca.encode(), mrp_ucca_fixed_back.encode()))


def test_ucca_tree_pipe(mrp_ucca, ucca, snt_token, tok_anchors):
    # need a step transform tree to graph by adding missing reentrance edges
    # step 1: tree to graph
    # if tree has top node, then the graph will also have
    graph = ucca.graph
    tree = ucca.tree
    if ucca.tree == None:
        logger.info("tree is None".format(mrp_ucca.id))
        return None
    else:
        logger.info("tree linearize: {}".format(tree.linearize()))
    new_graph,root_id = ptb2ucca(tree, snt_token, tok_anchors)
    # some me will be raised to make it connected
    #logger.error("new_graph after ptb2graph: for {}\n graph_nodes:{}\n  graph_edges:{}\n\n".format(mrp_ucca.id, new_graph.nodes.data(), new_graph.edges.data()))
    # step 2: adding corefence to graph
    # passed, now adding gold corefence here.
    for (h_role_align, d_role_align, edge) in ucca.reents:
        h_node = find_node_by_role_aligns(new_graph, h_role_align)
        d_node = find_node_by_role_aligns(new_graph, d_role_align)
        if h_node != None and d_node != None:
            if "attributes" in edge:
                new_graph.add_edge(h_node, d_node, key=edge["role"], role=edge["role"], attributes=edge["attributes"], values=edge["values"])
                new_graph.add_edge(d_node, h_node, key=edge["role"]+"-of", role=edge["role"]+"-of", attributes=edge["attributes"], values=edge["values"])
            else:
                new_graph.add_edge(h_node, d_node, key=edge["role"], role=edge["role"])
                new_graph.add_edge(d_node, h_node, key=edge["role"]+"-of", role=edge["role"]+"-of")
        else:
            logger.error("missing some reentrance :{}".format((h_role_align, d_role_align, edge)))

    for (h_role_align, d_role_align, ori_h_role_align) in ucca.fixed:
        d_role = d_role_align[0][1]
        ori_h_node = find_node_by_role_aligns(new_graph, ori_h_role_align)
        h_node = find_node_by_role_aligns(new_graph, h_role_align)
        d_node = find_node_by_role_aligns(new_graph, d_role_align)
        try:
            if ori_h_node != None and h_node != None and d_node != None:
                new_graph.remove_edge(ori_h_node, d_node)
                new_graph.remove_edge(d_node, ori_h_node)
                new_graph.add_edge(h_node, d_node, key=d_role, role=d_role)
                new_graph.add_edge(d_node, h_node, key=d_role+"-of", role=d_role+"-of")
            else:
                logger.error("missing some discontinuous:{},{}".format((h_role_align, d_role_align, ori_h_role_align), (h_node, d_node, ori_h_node)))
        except:
            logger.error(traceback.format_exc())
            logger.error("error happend when fix discontinuous nodes in {}, {}".format((h_role_align, d_role_align, ori_h_role_align), (ori_h_node, h_node, d_node)))

    #logger.error("new_graph after adding reentrance and fixed: for {}\n graph_nodes:{}\n  graph_edges:{}\n\n".format(mrp_ucca.id, new_graph.nodes.data(), new_graph.edges.data()))
    mrp_ucca_tree_back, M = UCCADecoder.graph_to_mrpGraph(mrp_ucca.id, new_graph, flavor=2, framework="ucca")
    mrp_ucca_tree_back.add_input(mrp_ucca.input)
    new_result = mces.evaluate([mrp_ucca], [mrp_ucca_tree_back], trace = 2);
    if new_result["all"]["f"] != 1.0:
        # logger.error("graph_nodes:{}\n  graph_edges:{}\n".format(graph.nodes.data(), graph.edges.data()))
        logger.error("new_graph: result is {} for {}\n graph_nodes:{}\n  graph_edges:{}\n gold:{}\n pred:{} \n\n".format(new_result, mrp_ucca.id, M.nodes.data(), M.edges.data(), mrp_ucca.encode(), mrp_ucca_tree_back.encode()))

def test_ucca_graph_pipe(mrp_ucca, ucca):
    # the original graph is also changed for splitting
    graph = ucca.graph
    h_v = BOS_WORD
    root_v = ucca.root
    mrp_ucca_back, _ = UCCADecoder.graph_to_mrpGraph(mrp_ucca.id, graph, flavor=2, framework="ucca")
    mrp_ucca_back.add_input(mrp_ucca.input)
    #result = mces.evaluate([mrp_ucca], [mrp_ucca], trace = 2);
    result = mces.evaluate([mrp_ucca], [mrp_ucca_back], trace = 2);
    if result["all"]["f"] != 1.0:
        logger.error("original_graph: result is {} for {}\n grpah_nodes:{}\n  graph_edges:{}\n gold:{}\n pred:{} \n\n".format(result,mrp_ucca.id, graph.nodes.data(), graph.edges.data(), mrp_ucca.encode(), mrp_ucca_back.encode()))


def test_ucca_pipe(data, n):
    if "ucca_matrix" in data:
        ucca_t = data["ucca_matrix"]
    else:
        ucca_t = None

    if "mrp_ucca" in data:
        mrp_ucca = data["mrp_ucca"]
    else:
        mrp_ucca = None

    #if mrp_ucca.id != "615001":
    #    return None
    logger.info("processing {}".format(mrp_ucca.id))
    snt_tokens = data["tok"]
    pos_tokens = data["pos"]
    tok_anchors = data['anchors']
    # gold uccaGraph
    ucca = UCCAGraph(ucca_t, mrp_ucca, snt_tokens, tok_anchors, pos_tokens)
    #test_ucca_graph_pipe(mrp_ucca, ucca)
    #test_ucca_tree_pipe(mrp_ucca, ucca, snt_tokens, tok_anchors)
    #test_ucca_fixed_pipe(mrp_ucca, ucca, snt_token, tok_anchors)
    return ucca

def test_mrp_dataset(dataset, output_file):
    n = 0
    m = 0
    with open(output_file,"w+") as ofile:
        for id, data in dataset.items():
            # try to extend this to other frameworks
            if 'mrp_ucca' in data:
                n = n + 1
                if n % 500 ==0:
                    logger.info(n)
                ucca=test_ucca_pipe(data, n)
                if ucca and ucca.tree:
                    m = m +1
                    ofile.write(ucca.tree.linearize())
                    ofile.write("\n")
    logger.info("{} valid tree written down in {}".format(m, n))


def test_mrp_to_graph():
    logger.info("processing training set")
    training_data = readFeaturesInput(opt.trainingCompanionFilesPath)
    mergeWithAnnotatedGraphs(training_data, opt.trainingFilesPath)
    test_mrp_dataset(training_data, opt.trainingTreePath)

    logger.info(("processing development set"))
    dev_data = readFeaturesInput(opt.devCompanionFilesPath)
    mergeWithAnnotatedGraphs(dev_data, opt.devFilesPath)
    test_mrp_dataset(dev_data, opt.devTreePath)

    logger.info("processing test set")
    test_data = readFeaturesInput(opt.testCompanionFilesPath)
    mergeWithAnnotatedGraphs(test_data, opt.testFilesPath)
    test_mrp_dataset(test_data, opt.testTreePath)


def main(opt):
    test_mrp_to_graph()

if __name__ == "__main__":
    global opt
    parser = arg_parser()
    opt = parser.parse_args()
    suffix = opt.suffix
    opt.trainFolderPath = opt.build_folder + "/training/"
    opt.trainingFilesPath = folder_to_files_path(opt.trainFolderPath, suffix)
    opt.trainingTreePath = opt.trainFolderPath +"/training.ptb_tree"
    opt.trainingCompanionFilesPath = folder_to_files_path(opt.trainFolderPath, opt.companion_suffix)

    opt.devFolderPath = opt.build_folder + "/dev/"
    opt.devFilesPath = folder_to_files_path(opt.devFolderPath, suffix)
    opt.devTreePath= opt.devFolderPath +"/dev.ptb_tree"
    opt.devCompanionFilesPath = folder_to_files_path(opt.devFolderPath, opt.companion_suffix)

    opt.testFolderPath = opt.build_folder + "/test/"
    opt.testFilesPath = folder_to_files_path(opt.testFolderPath, suffix)
    opt.testTreePath = opt.testFolderPath +"/test.ptb_tree"
    opt.testCompanionFilesPath = folder_to_files_path(opt.testFolderPath, opt.companion_suffix)

    logger.info("opt:{}".format(opt))

    main(opt)
