#!/usr/bin/env python3.6
# coding=utf-8
'''

DMGraph representing DM graph as networkx graph,
Being able to apply recategorization to original graph,
which involves collapsing nodes for concept identification and unpacking for relation identification.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-28

make DMGraph as a proxy for DM parsing, by offering a construct to transform a MRP graoh in DMGraph
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-06-07
'''
from utility.constants import *
import networkx as nx
import logging
import json

logger = logging.getLogger("mrp.utility.dm_utils.DMGraph")

class DMGraph(object):
    def __init__(self, anno, mrp_graph = None, normalize_inverses=True,
                 normalize_mod=True, tokens=None,single_edge_only=True):
        '''
        make a DM matrix anno or mrp_graph into our Graph proxy with networkx
        '''
        # networkx graph structure
        self.normalize_inverses = normalize_inverses
        self.normalize_mod = normalize_mod
        self.graph = nx.MultiDiGraph()
        self.single_edge_only = single_edge_only
        if anno:
            # dm , _anno, is a matrix [['A','B','C',],['D','E','F']]
            self._anno = anno
            # parse the original DM matrix into a graph
            self.mrp_graph = matrix2graph(self._anno, framework = "dm", text = None)
            if g is None:
                raise DMForamtError('Well-formedness error in annotation:\n' + matrix2string(self._anno())+"\n")
            self._analyze_mrp_graph(self.mrp_graph)
            self.id = self.mrp_graph.id
        elif mrp_graph:
            # load mrp_graph into DMGraph
            # when loading from mrp_graph, there is no more parsimonious info
            # 1. transform every node in mrp_graph.nodes into self.nodes, with its id as variable name
            self.mrp_graph = mrp_graph
            self._analyze_mrp_graph(self.mrp_graph)
            self._anno = json.dumps(self.mrp_graph.encode())
            self.id = self.mrp_graph.id
        else:
            raise NotImplementedError("Both anno(DM Matrix) and mrp_graph is NONE")

    def _analyze_mrp_graph(self, g):
        '''
        Analyze the MRP graph produced by MRP Graph, make it into a specific DM Graph struct
        '''
        roots=[]
        if len(g.nodes) == 0:
            self.root = None
            logger.warn("empty graph g:{}".format(g.id))
            return

        for node in g.nodes:
            v = id2Var(node.id)
            node_v = DMUniversal(mrp_node=node)
            # here we still use anchors, without mapping into tokens
            self.graph.add_node(v, value=node_v, anchors=node.anchors, gold=True)
            if node.is_top:
                roots.append(v)
        # cat training.mrp_dm | grep -oP "\"tops\": \[\d+\]" | grep ","
        # now it seems only one top nodes for each DM, to be verified
        # now only consider the first top node as gold top
        if len(roots) > 0:
            self.root = roots[0]
            if len(roots) != 1:
                logger.error("DM {} should have a single top nodes".format(g.id))
        else:
            # self.root = list(self.graph.nodes)[0]
            self.root = None   # not permit none root for now

        for edge in g.edges:
            h = id2Var(edge.src)
            h_v = self.graph.nodes[h]
            d = id2Var(edge.tgt)
            d_v = self.graph.nodes[d]
            # we forece add ":" in front of the edge label for consistent with AMR
            r = ":"+edge.lab
            # in DM, there is no inversed edges.
            if self.single_edge_only and d in self.graph[h]:
                logger.info("{},\n single_edge_only={}, multi_edges:{} and {}".format(self._anno, self.single_edge_only, str(self.graph[h][d]), (h_v, r,d_v)))
                continue
            else:
                self.graph.add_edge(h, d, key=r, role=r)
                # here we also adding the inversed relation for the connectivity for DiGraph
                self.graph.add_edge(d, h, key=r+"-of",role=r + "-of")

    def get_gold(self):
        """
        for a DMGraph, return all the gold concept and roles.
        """
        cons = []
        roles = []
        for n, d in self.graph.nodes(True):
            # add gold concepts into a list
            if "gold" in d:
                v = d["value"]
                cons.append(v)

        for h, d, _, rel in self.graph.edges(keys=True,data=True):
            # add roles into a list, every role is [h, d, r]
            r = rel["role"]
            # during learning, only predict the cannonical edge labels, its inversed version is only for connectivity of DiGraph
            if self.cannonical(r):
                assert "gold" in self.graph.node[h] and "gold" in self.graph.node[d]
                h = self.graph.node[h]["value"]
                d = self.graph.node[d]["value"]
                roles.append([h,d,r])

        if self.root:
            root = self.graph.node[self.root]["value"]
            # todo: add a special Node for DM
            roles.append([DMUniversal.TOP_DMUniversal(),root,':top'])

        # WARN: here roles may not contains any top relations
        return cons,roles

    def __getitem__(self, item):
        return self.graph.node[item]

    #check whether the relation is in the cannonical direction
    # ARG0, ARG1, ... ARGn as core rel
    # BV also as core rel.
    # compound and mwe are special relation, which are usually happened in consecutive tokens.
    # Now also make the model to learn this.
    def cannonical(self,r):
        # now all rel are core, and all forward are cannonical
        return ("-of"  not in r and  self.is_core(r))

    @staticmethod
    def is_core(r):
        """
        for DM, now treat all edge as core rel
        """
        return ("-of" not in r)

    @staticmethod
    def is_inversed_edge(role):
        return role.endswith("-of")

    @staticmethod
    def is_inversed_edge(edge):
        if edge.endswith("-of"):
            return True
        else:
            return False

    @staticmethod
    def get_inversed_edge(edge):
        if edge.endswith("-of"):
            inverse = edge[:-3]
        else:
            inverse = edge + "-of"
        return inverse

    @staticmethod
    def get_normalizede_edge(edge):
        if is_inversed_edge(edge):
            return get_inversed_edge(edge)
        else:
            return edge

    def getRoles(self,node,index_dict,rel_index,relyed = None):
        """
        get all the roles of a node
        node : node variable
        index_dict is dict(key=node, value = intIndex), the index is the index for recategorized nodes
        rel_index, is dict(key=node, value = intIndex), the index is the index for gold nodes
        return [[node, node1ReCateIndex], [[rel, node2GoldIndex]]]]
        """
        # (amruniversal,index,[[role,rel_index]])
        if relyed and relyed not in index_dict:
            print ("rely",node,relyed,self.graph.node[relyed]["value"],index_dict,self._anno)
        elif relyed is None and node not in index_dict: print (self.graph.node[node]["value"])
        # get only the original node index, this index is recategorised index
        index = index_dict[node] if relyed is None else index_dict[relyed]
        out = []
        #   if self.graph.node[node]["value"].le != "name":
        # self.graph[node] returns all the adj node in a dict(key=neighbor, value=attributes)
        for n2 in  self.graph[node]:
            # the role from n2 to node
            for key, edge_data in self.graph[node][n2].items():
                r = edge_data["role"]
                if self.cannonical(r):
                    if n2 not in rel_index:
                        print(self._anno)
                    # out is [rel_role, gold_dep_node_id]
                    out.append([r,rel_index[n2]])
        return [[self.graph.node[node]["value"],index], out]

    #return data for training concept identification or relation identification
    def node_value(self, keys=["value"], all=False):
        def concept_concept():
            """
            out: all nodes after recategorizing
            index_dict, [key:node, value: index], index is the order for transduce the node in the AMR graph.
            """
            # out is an array, [[subnode1, subnode-attr], [subnode2, subnode]]
            out = []
            # index the order id of a node.
            index = 0
            # save a node2index dict
            index_dict ={}
            for n, d in self.graph.nodes(True):
                # https://networkx.github.io/documentation/networkx-2.1/reference/classes/generated/networkx.Graph.nodes.html
                # n is the node, d is the data with all attributes
                # graph.nodes(True), means return entire node attribute dict
                # if it has recategorized new nodes, iterate its recategorizeed nodes, only add the combination nodes, not the original node
                if "original-of" in d:
                    comps = d["original-of"]
                    for comp in comps:
                        if comp is None:
                            continue
                        comp_d = self.graph.node[comp]
                        # output a (node, value1, value2)
                        # by default key is value, which is AMRUniversal Node of that node.
                        out.append([comp] + [comp_d[k] for k in keys])
                        index_dict[comp] = index
                        index += 1
                elif not ("has-original" in d or  "rely" in d):
                    # TODO: all node in DM is the original node, without categorizing
                    # not a recategorized node, just use that node itself.
                    out.append([n] + [d[k] for k in keys])
                    index_dict[n] = index
                    index += 1
            # out is an array, [[subnode1, subnode-attr], [subnode2, subnode2_attr]]
            # index_dict is dict(key=node, value = intIndex)
            return out,index_dict

        def rel_concept():
            """
            return the gold node and its node2index dict
            """
            index = 0
            rel_index ={}
            # rel_out is in shape like [[n, d]]
            rel_out = []
            # If True, return entire node attribute dict as (n, ddict).
            # n is node Varible, d is all the dict attributes, index is nodes index.
            for n, d in self.graph.nodes(True):
                if "gold" in d:
                    rel_out.append([n,d])
                    rel_index[n] = index
                    index += 1

            # rel_out is in shape like [[var, Node]]
            # rel_index is dict(key = var, value= Int index)
            return rel_out,rel_index

        # out: all the nodes after recategorization
        # index_dict, [key:node, value: index], index is the order for transduce the node in the AMR graph.
        out,index_dict = concept_concept()
        if all:
            # all means all attributes
            # rel_out: all the gold concepts
            # rel_index: a different index from gold node transduce order.
            rel_out, rel_index = rel_concept()
            for i, n_d in enumerate( rel_out):
                n,d = n_d
                # rely means n is a original node
                if "rely" in d:
                    # [[node, node1ReCateIndex], [[rel, node2GoldIndex]]]]
                    rel_out[i] = self.getRoles(n,index_dict,rel_index,d["rely"])
                elif not ("has-original" in d or  "original-of" in d):
                    # it is an original node
                    # DM will follow this path
                    rel_out[i] = self.getRoles(n,index_dict,rel_index)
                else:
                    # gold node should not have a recategorized node
                    assert False , (self._anno, n, d["value"])

            if self.root:
                # if there is root, then root index must be in the roots.
                assert (self.root in rel_index),(self.graph.nodes[self.root],rel_index,self._anno)
                root_index = rel_index[self.root]
            else:
                root_index = None
            # only return the gold concepts, and it expanded nodes.
            # out : all the concept nodes include the recategorized one, include the top node, recatgorized nodes
            # out is an array, [[subnode1, subnode-attr], [subnode2, subnode]]
            # rel_out: all the original gold concepts, [[node, node_attr]] for every node, list its head and dependent,  [[head, headIndex], [[rel, depIndex]]]]
            # rel_index:  store the index of the node in the order of gold amr nodes
            return out,rel_out,root_index
        else:
            # return all predicted concepts
            return out

class DMFormatError(Exception):
    pass

def matrix2string(matrix):
    return '\n'.join(['\t'.join(row) for row in matrix])

def id2Var(id):
    return DMVar("id"+str(id))

class DMVar(object):
    """
    In this AMR file, nodes are classified into 3 classes: Concept, Var, Constant
    The variable class, used for variable for a name, representing the reentranies.
    """
    def __init__(self, name):
        self._name = name

    # simply override the less equal than
    def __le__(self,other):
        return self._name < other._name

    def is_var(self):
        return True

    def is_concept(self):
        return False

    def is_constant(self):
        return False

    def __repr__(self):
        return 'DMVar(' + self._name +')'

    # override the string into its name, name is the identity of the variable node.
    def __str__(self):
        return self._name

    def __call__(self, **kwargs):
        return self.__str__()

    def __eq__(self, that):
        return isinstance(that, type(self)) and self._name == that._name

    def __hash__(self):
        return hash(self._name)

    def deepcopy(self,s=""):
        return DMVar(self._name+s)

#decompose dm node to le_pos_cat_sense
def decompose(c):
    """
    extract le, pos and cat from a single dm node
    """
    if c is None: return None,None,None,None
    if isinstance(c,DMUniversal):
        return c.le,c.pos,c.cat,c.sense
    return None, None, None, None

class DMUniversal(object):
    def __init__(self, *args, **kwargs):
        # *args for un-named argeument, kwarges are keyworded arguments
        if "string" in kwargs:
            raise NotImplementedError("construact from string is not supported on DM")
        elif "mrp_node" in kwargs:
            self.construct_by_mrp_node(kwargs["mrp_node"])
        else:
            self.construct_by_content(args[0], args[1],args[2], args[3], args[4])

    @staticmethod
    def TOP_DMUniversal():
        return DMUniversal(BOS_WORD,BOS_WORD, BOS_WORD, BOS_WORD, None)

    @staticmethod
    def NULL_DMUniversal():
        return DMUniversal(NULL_WORD,NULL_WORD, NULL_WORD, NULL_WORD, None)

    def construct_by_mrp_node(self, node):
        self.le = node.label
        try:
            i = node.properties.index("pos");
            self.pos = node.values[i]
        except:
            logger.error("no pos properties in node:{}".format(json.dumps(node.encode())))
            self.pos = NULL_WORD

        try:
            i = node.properties.index("frame");
            frame=node.values[i]
            self.cat=frame.split(':')[0]
            self.sense=frame.split(':')[1]
        except:
            logger.error("no frame properties in node:{}".format(json.dumps(node.encode())))
            self.cat=NULL_WORD
            self.sense=NULL_WORD

        if node.anchors:
            self.anchors = node.anchors
        else:
            self.anchors = None


    def construct_by_content(self, pos, cat, sense, le = None, anchors=None):
        """
        NULL_WORD is empty string, which is just place holder but not break the structure a node
        """
        self.le = le
        self.pos = pos
        # for dm, we split the original frame into two parts, the part before ‘:’
        # _{lemma}{cat}_: sense will find the lexicon entry, most of the time, it is unique.
        self.cat = cat
        self.sense = sense
        self.anchors = anchors

    def get_frame(self):
        if self.sense is None:
            return self.cat
        else:
            return self.cat +":" + self.sense

    def __str__(self):
        return self.__repr__()

    def no_anchor_copy(self):
        return DMUniversal(self.pos,self.cat, self.sense, self.le, None)

    def to_tuple(self):
        if self.anchors:
            anchors_str = ",".join(["{}-{}".format(x["from"],x["to"]) for x in self.anchors])
        else:
            anchors_str = "N/A"
        return (self.le,self.pos,self.cat,self.sense, anchors_str)

    def get_anchors_str(self):
        if self.anchors:
            return ",".join(["{}-{}".format(x["from"],x["to"]) for x in self.anchors])
        else:
            return " - "

    def __repr__(self):
        return "({},{},{},{})-({})".format(self.le, self.pos, self.cat, self.sense, self.get_anchors_str())

    def __hash__(self):
        return hash(self.__repr__())

    def non_sense_equal(self,other):
        return isinstance(other,DMUniversal) and self.le == other.le and self.pos == other.pos and self.cat == other.cat and self.anchors == other.anchors

    def non_anchors_equal(self,other):
        return isinstance(other,DMUniversal) and self.le == other.le and self.pos == other.pos and self.cat == other.cat and self.sense == other.sense

    def __eq__(self, other):
        return isinstance(other, DMUniversal) and self.le == other.le and self.pos == other.pos and self.cat ==other.cat and self.sense == other.sense and self.anchors == other.anchors
