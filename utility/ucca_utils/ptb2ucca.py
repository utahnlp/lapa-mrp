#!/usr/bin/env python3
# coding=utf-8
'''

Load amr transformed mrp, and load into AMRGraph as gold label, test the output with the gold AMRGraph
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-06-23
'''

from parser.UCCAProcessors import *
import os
import sys
from torch import cuda
from utility.ucca_utils.UCCAGraph import *
from utility.ptb_utils.trees import *
from utility.data_helper import folder_to_files_path, load_top_dataset
from src import *
from parser.Dict import read_dicts
from src.config_reader import get_parser
from utility.mtool.score import mces
from utility.constants import *

logger = logging.getLogger("mrp.ucca")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def ptb2ucca(tree, tokens, tok_anchors):
    """
    transform a ptb to graph, without adding more extra corefenece nodes
    without TOP_NODE
    """
    nid = 0

    def get_anchors(treeNode, tokens, tok_anchors, tok_nid):
        if isinstance(treeNode, InternalTreebankNode):
            return None, None
        else:
            if tok_nid < len(tokens) and tokens[tok_nid] == treeNode.word:
                return tok_anchors[tok_nid], [tok_nid]

            words = treeNode.word.split()
            #logger.info("unaligned treeNode {}".format(treeNode))
            # need unescape
            anchors = []
            aligns= []
            for word in words:
                if word in PTB_TOKEN_UNESCAPE:
                    word = PTB_TOKEN_UNESCAPE[word]
                for i, (token, anchor) in enumerate(zip(tokens, tok_anchors)):
                    if word == token:
                        if abs(i - tok_nid) < 3:
                            anchors.extend(anchor)
                            aligns.append(i)
            if len(aligns) == 0:
                #logger.info("tree unaligned , {}, {}, {},{}".format(treeNode.word, tok_nid,  tokens, tok_anchors))
                pass
            return anchors, aligns

    def fix_aligns(graph):
        def fix_helper(parent):
            children = []
            # sort the edges from according the alignments of dep node
            # logger.info("edges: {}".format(graph[parent].items()))
            sorted_children = sorted(list(graph[parent].items()), key=lambda x: min(graph.node[x[0]]['align']) if 'align' in graph.node[x[0]] and graph.node[x[0]]['align'] !=None and len(graph.node[x[0]]['align']) > 0 else sys.maxsize)

            # logger.info("parent: {}, children:{}".format(parent, sorted_children))
            for c,_ in sorted_children:
                # logger.info("c: {}, parent: {}".format(c, parent))
                for key, edge_data in graph[parent][c].items():
                    if edge_data["role"].endswith("-of"):
                        continue
                    else:
                        if c in used_nodes:
                            continue
                        else:
                            used_nodes.append(c)

                        if 'align' not in graph.node[c] or graph.node[c]['align'] == None:
                            # non-terminal nodes, in a tree, it is continuous
                            sub_children = fix_helper(c)
                            graph.node[c]['align'] = sub_children
                            children.extend(sub_children)
                        else:
                            # terminal nodes
                            for i in graph.node[c]['align']:
                                children.append(i)
            # sort and unique children
            return sorted(set(children))

        used_nodes = [TOP_NODE]
        graph.node[TOP_NODE]['align']=fix_helper(TOP_NODE)

    def helper(parent):
        parent_nid = len(graph.nodes)
        if isinstance(parent, InternalTreebankNode):
            ntype = INTERNAL_NODE
            uni = UCCAUniversal(ntype, None)
            graph.add_node(parent_nid, value=uni, anchors=None, align=None)
        else:
            ntype = LEAF_NODE
            tok_nid = len(tok_nodes)
            n_anchors, n_aligns = get_anchors(parent, tokens, tok_anchors, tok_nid)
            uni = UCCAUniversal(ntype, n_anchors)
            graph.add_node(parent_nid, value=uni, anchors=n_anchors, align=n_aligns)
            tok_nodes.append(parent_nid)
        # logger.info("n_id: {}".format(parent_nid))
        # add children
        if isinstance(parent, InternalTreebankNode):
            leaf_tok_nodes = []
            for child in parent.children:
                # add edge between parent to its child
                child_nid = helper(child)
                if isinstance(child, InternalTreebankNode):
                    graph.add_edge(parent_nid, child_nid, key=child.label, role=child.label)
                    graph.add_edge(child_nid, parent_nid, key=child.label+"-of", role=child.label+"-of")
                else:
                    graph.add_edge(parent_nid, child_nid, key=child.tag, role=child.tag)
                    graph.add_edge(child_nid, parent_nid, key=child.tag+"-of", role=child.tag+"-of")
                    if UCCAGraph.is_tok_edge(child.tag) and not UCCAGraph.is_inversed_edge(child.tag):
                        leaf_tok_nodes.append(child_nid)
            if len(leaf_tok_nodes)  == len(parent.children) and len(set(leaf_tok_nodes)) > 1:
                # multiple edges within two nodes, duplicate children existed
                # all children are leaf tok nodes, merge them
                multiword_nodes.append((parent_nid, leaf_tok_nodes))
        else:
            # leave node, pass
            pass
        return parent_nid

    graph = nx.MultiDiGraph()
    tok_nodes = []
    multiword_nodes = []
    # tree keep the leaf order as the tokens
    root_id = helper(tree)
    graph.add_node(TOP_NODE, value=UCCAUniversal.TOP_UCCAUniversal(),anchors=None, align=None)
    graph.add_edge(TOP_NODE, root_id, key='TOP', role='TOP')
    graph.add_edge(root_id, TOP_NODE, key='TOP-of', role='TOP-of')
    #logger.error("new_graph inital tree :graph_nodes:{}\n  graph_edges:{}\n\n".format(graph.nodes.data(), graph.edges.data()))
    fix_aligns(graph)

    #logger.error("new_graph after fix aligns: graph_nodes:{}\n  graph_edges:{}\n\n".format(graph.nodes.data(), graph.edges.data()))
    # step 2: adding corefence to graph
    # contract the nodes, especially the mwe nodes
    H = graph.copy()
    for parent_nid, nodes in multiword_nodes:
        x = nodes[0]
        anchors = H.node[x]['anchors'] if H.node[x]['anchors'] else []
        align = H.node[x]['align'] if H.node[x]['align'] else []
        for y in nodes[1:]:
            for anchor in graph.node[y]['anchors']:
                in_x = False
                for anchor_x in anchors:
                    if anchor['from'] == anchor_x['from'] and anchor['to'] == anchor_x['to']:
                        in_x = True
                if not in_x:
                    anchors.append(anchor)
            for a in graph.node[y]['align']:
                if a not in align:
                    align.append(a)
            # key is not trusted to be always the role
            # after contracting, the  key will be numbered instead of the role
            # before contracning, we need to remove other TOK_TAG from its parent to them
            H.remove_edge(parent_nid, y)
            H.remove_edge(y, parent_nid)
            H = nx.contracted_nodes(H, x, y, self_loops=False)
        new_x_uni = UCCAUniversal(LEAF_NODE, anchors)
        H.node[x]['value'] = new_x_uni
        H.node[x]['anchors'] = anchors
        H.node[x]['align'] = sorted(align)

    # contract edges for TOK
    M = H.copy()
    for h, d, key, edge_data in H.edges(keys=True, data=True):
        if UCCAGraph.is_unk_edge(key) or (UCCAGraph.is_tok_edge(key) and not UCCAGraph.is_inversed_edge(key)):
            # logger.error(" contracted {}".format((h, d, key, edge_data)))
            # d will be ke, words are contracted into its parent
            anchors = M.node[h]['anchors'] if M.node[h]['anchors'] else []
            align = M.node[h]['align'] if M.node[h]['align'] else []
            for anchor in H.node[d]['anchors']:
                in_h = False
                for anchor_h in anchors:
                    if anchor['from'] == anchor_h['from'] and anchor['to'] == anchor_h['to']:
                        in_h = True
                if not in_h:
                    anchors.append(anchor)
            for a in H.node[d]['align']:
                if a not in align:
                    align.append(a)
            M = nx.contracted_nodes(M, h, d, self_loops=False)
            new_x_uni = UCCAUniversal(LEAF_NODE, anchors)
            M.node[h]['value'] = new_x_uni
            M.node[h]['anchors'] = anchors
            M.node[h]['align'] = sorted(align)
        elif (not str(key).startswith(":")) and (not str(key).startswith('TOP')):
            logger.error(" contracted edge should not exist {}".format((h, d, key, edge_data)))

    return M,root_id

def find_node_by_role_aligns(graph, role_aligns):
    # this must be done when all aligns has been aassigned
    (p_aligns, in_role), aligns = role_aligns
    #logger.info("in_role:{}, aligns:{}".format(in_role, aligns))
    x = None
    for h, d, key, edge_data in graph.edges(keys=True, data=True):
        if graph.node[d]['align'] == aligns and edge_data["role"] == in_role and graph.node[h]['align'] == p_aligns:
            x = d

    if x == None:
        for h, d, key, edge_data in graph.edges(keys=True, data=True):
            if graph.node[d]['align'] == aligns and edge_data["role"] == in_role:
                x = d

    return


def arg_parser():
    parser = argparse.ArgumentParser(description='p2bucca.py')

    ## Data options
    parser.add_argument('--input_ptb', default="", type=str,
                        help="""input_ptb_file to ucca""")
    parser.add_argument('--input', default="", type=str,
                        help="""input_conllu_file""")
    parser.add_argument('--extract_snt', default="", type=str,
                        help="""output snt file""")
    return parser

def test_ucca_dataset(fex_list, tree_list, output_file):
    n = 0
    with open(output_file,"w+") as ofile:
        for fex, tree in zip(fex_list, tree_list):
            n = n + 1
            if n % 500 ==0:
                logger.info(n)

            snt_tokens = fex["tok"]
            tok_anchors = fex["anchors"]
            nx_graph,root_id = ptb2ucca(tree, snt_tokens, tok_anchors)
            mrp_graph, M = UCCADecoder.graph_to_mrpGraph(fex["example_id"], nx_graph, flavor=2, framework="ucca", sentence=fex['input_snt'])
            # for reentrance and fixed
            ofile.write(json.dumps(mrp_graph.encode(), indent=None, ensure_ascii = False))
            ofile.write("\n")


def test_ucca_snt(fex_list, output_file):
    n = 0
    with open(output_file,"w+") as snt_out:
        for fex in fex_list:
            n = n + 1
            if n % 500 ==0:
                logger.info(n)
            snt_tokens = fex["tok"]
            snt_out.write(" ".join(snt_tokens))
            snt_out.write("\n")


def test_ptb2ucca():
    logger.info("processing input fex{}".format([opt.input]))
    fex_list = readFeaturesInputList([opt.input])
    test_ucca_snt(fex_list, opt.extract_snt)
    if opt.input_ptb:
        logger.info("processing input ptb{}".format(opt.input_ptb))
        tree_list = load_trees(opt.input_ptb, strip_top=False)
        #assert len(fex_list) == len(tree_list), "inconsistency length between fex_list {} and tree_list {}".format(len(fex_list), len(tree_list))
        test_ucca_dataset(fex_list, tree_list, opt.output_ucca)

def main(opt):
    test_ptb2ucca()

if __name__ == "__main__":
    global opt
    parser = arg_parser()
    opt = parser.parse_args()
    opt.output_ucca = opt.input_ptb+".ucca"
    logger.info("opt:{}".format(opt))

    main(opt)
