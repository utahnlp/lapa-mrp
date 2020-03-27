#!/usr/bin/env python3.6
# coding=utf-8
'''

PSDGraph representing PSD graph as networkx graph,
Being able to apply recategorization to original graph,
which involves collapsing nodes for concept identification and unpacking for relation identification.

make PSDGraph as a proxy for PSD parsing, by offering a construct to transform a MRP graoh in PSDGraph
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-06-07
'''
from utility.constants import *
from utility.psd_utils.VallexReader import *
import networkx as nx
import logging
import json

logger = logging.getLogger("mrp.utility.psd_utils.PSDGraph")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class PSDGraph(object):
    def __init__(self, anno, mrp_graph = None, normalize_inverses=True,
                 normalize_mod=True, tokens=None,single_edge_only=True):
        '''
        make a PSD matrix anno or mrp_graph into our Graph proxy with networkx
        '''
        # networkx graph structure
        self.normalize_inverses = normalize_inverses
        self.normalize_mod = normalize_mod
        self.graph = nx.MultiDiGraph()
        self.single_edge_only = single_edge_only
        if anno:
            # psd , _anno, is a matrix [['A','B','C',],['D','E','F']]
            self._anno = anno
            # parse the original PSD matrix into a graph
            self.mrp_graph = matrix2graph(self._anno, framework = "psd", text = None)
            if g is None:
                raise PSDForamtError('Well-formedness error in annotation:\n' + matrix2string(self._anno())+"\n")
            self._analyze_mrp_graph(self.mrp_graph)
            self.id = self.mrp_graph.id
        elif mrp_graph:
            # load mrp_graph into PSDGraph
            # when loading from mrp_graph, there is no more parsimonious info
            # 1. transform every node in mrp_graph.nodes into self.nodes, with its id as variable name
            self.mrp_graph = mrp_graph
            self._analyze_mrp_graph(self.mrp_graph)
            self._anno = json.dumps(self.mrp_graph.encode())
            self.id = self.mrp_graph.id
        else:
            raise NotImplementedError("Both anno(PSD Matrix) and mrp_graph is NONE")

    def _analyze_mrp_graph(self, g):
        '''
        Analyze the MRP graph produced by MRP Graph, make it into a specific PSD Graph struct
        '''
        roots=[]
        if len(g.nodes) == 0:
            self.root = None
            logger.warn("empty graph g:{}".format(g.id))
            return

        for node in g.nodes:
            v = id2Var(node.id)
            node_v = PSDUniversal(mrp_node=node)
            # here we still use anchors, without mapping into tokens
            self.graph.add_node(v, value=node_v, anchors=node.anchors, gold=True)
            if node.is_top:
                roots.append(v)
        # cat training.mrp_psd | grep -oP "\"tops\": \[\d+\]" | grep ","
        # now it seems only one top nodes for each PSD, to be verified
        # now only consider the first top node as gold top
        if len(roots) > 0:
            self.root = roots[0]
            if len(roots) != 1:
                logger.error("PSD {} should have a single top nodes".format(g.id))
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
            # in PSD, there is no inversed edges.
            if self.single_edge_only and d in self.graph[h]:
                logger.info("{},\n single_edge_only={}, multi_edges:{} and {}".format(self._anno, self.single_edge_only, str(self.graph[h][d]), (h_v, r,d_v)))
                continue
            else:
                #self.graph.add_edge(h, d, key=r, role=r)
                #self.graph.add_edge(d, h, key=r+"-of",role=r + "-of")

                # there is no -of or labels deleting -arg
                if r.endswith('-arg'):
                    # it is not gold, it is simplifed version for prediction
                    if PSDGraph.is_must_arg_functor(r):
                        # keep args, it will be canonical
                        self.graph.add_edge(h, d, key=r, role=r, cls=True)
                        # here we also adding the inversed relation for the connectivity for DiGraph
                        # not cannonical
                        self.graph.add_edge(d, h, key=r+"-of",role=r + "-of", cls=True)
                    else:
                        if not PSDGraph.check_oblig_args_in_vallex(r[:-4], h_v["value"]):
                            logger.error("ARGError: {} is not oblig in {}, id= {}".format(r[:-4], h_v["value"], g.id))
                        # with args, and not must-args, it is not canonical
                        self.graph.add_edge(h, d, key=r, role=r)
                        # with args, and not must-args, backward is canonical, but it should not be used for classification
                        self.graph.add_edge(d, h, key=r+"-of",role=r + "-of")
                        # remove -args, forward is not canonical
                        self.graph.add_edge(h, d, key=r[:-4], role=r[:-4], reduced=True, cls=True)
                        # remove -args, backword is canonical
                        self.graph.add_edge(d, h, key=r[:-4]+"-of",role=r[:-4] + "-of", reduced=True, cls=True)
                else:
                    if PSDGraph.check_oblig_args_in_vallex(r, h_v["value"]):
                        logger.error("ARGError: {} is oblig in {}, but without -arg, id ={} ".format(r[:-4], h_v["value"], g.id))
                    # if is_core, forward is canonical, not if core, then is not cannotcial
                    self.graph.add_edge(h, d, key=r, role=r, cls=True)
                    # here we also adding the inversed relation for the connectivity for DiGraph
                    # if is_core, -of is not canonical, not core, then it is cannocial
                    self.graph.add_edge(d, h, key=r+"-of",role=r + "-of", cls=True)

    def get_gold(self):
        """
        for a PSDGraph, return all the gold concept and roles.
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
            if self.cannonical(r) and 'reduced' not in rel:
            #if self.cannonical(r):
                assert "gold" in self.graph.node[h] and "gold" in self.graph.node[d]
                h = self.graph.node[h]["value"]
                d = self.graph.node[d]["value"]
                roles.append([h,d,r])

        if self.root:
            root = self.graph.node[self.root]["value"]
            # todo: add a special Node for PSD
            roles.append([PSDUniversal.TOP_PSDUniversal(),root,':top'])

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
        """
        for PSD, make important node on a head for a relation
        only .member and four core args are forward, others are backwards
        """
        return  ("-of" in r and not self.is_core(r)) or ("-of"  not in r and  self.is_core(r))

    @staticmethod
    def check_oblig_args_in_vallex(r, uni):
        frame_id = uni.get_frame()
        return g_vallex_reader.check_functor_in_oblig_args(frame_id, r[1:])

    @staticmethod
    def is_core(r):
        """
        for PSD, -arg evoked by verb predicate, .member usually evoked by conjunction or punc
        For others are reversed, predicate will invoke more meaning, example time will invoke(":TOWH-of")
        when a predicate take argument with the above role, it will become x-arg
        """
        return r.startswith(':top') or ('.member' in r) or PSDGraph.is_must_arg_functor(r)

# any(r.startswith(x) for x in [':ACT', ':CPHR', ':DPHR', ':PAT', ':ADDR', ':ORIG', ':EFF', ':BEN', ':LOC', ':DIR1', ':DIR2', ':DIR3', ':TWHEN', ':TFRWH', ':TTILL', ':TOWH', ':TSIN', ':TFHL', ':MANN', ':MEANS', ':ACMP', ':EXT', ':INTT', ':MAT', ':APP', ':CRIT', ':REG'])

    @staticmethod
    def is_must_arg_functor(r):
        """
        must with -arg
        """
        return any(r.startswith(x) for x in [':ACT',':ADDR',':EFF',':ORIG',':PAT'])

    @staticmethod
    def is_arg_functor(r):
        return any(r.startswith(x) for x in [':ACT', ':CPHR', ':DPHR', ':PAT', ':ADDR', ':ORIG', ':EFF', ':BEN', ':LOC', ':DIR1', ':DIR2', ':DIR3', ':TWHEN', ':TFRWH', ':TTILL', ':TOWH', ':TSIN', ':TFHL', ':MANN', ':MEANS', ':ACMP', ':EXT', ':INTT', ':MAT', ':APP', ':CRIT', ':REG'])

    @staticmethod
    def is_inversed_edge(role):
        return role.endswith("-of")

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
                # only add no-arg relations for prediction
                if self.cannonical(r) and 'cls' in edge_data:
                #if self.cannonical(r):
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
                    # TODO: all node in PSD is the original node, without categorizing
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
                    # PSD will follow this path
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

class PSDFormatError(Exception):
    pass

def matrix2string(matrix):
    return '\n'.join(['\t'.join(row) for row in matrix])

def id2Var(id):
    return PSDVar("id"+str(id))

class PSDVar(object):
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
        return 'PSDVar(' + self._name +')'

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
        return PSDVar(self._name+s)

def decompose(c):
    """
    extract le, pos from a single psd node
    """
    if c is None: return None,None,None,None
    if isinstance(c,PSDUniversal):
        return c.le,c.pos,c.sense
    return None, None, None

class PSDUniversal(object):
    def __init__(self, *args, **kwargs):
        # *args for un-named argeument, kwarges are keyworded arguments
        if "string" in kwargs:
            raise NotImplementedError("construact from string is not supported on PSD")
        elif "mrp_node" in kwargs:
            self.construct_by_mrp_node(kwargs["mrp_node"])
        else:
            # pos, lemma, sense, anchors
            self.construct_by_content(args[0], args[1],args[2], args[3])

    @staticmethod
    def TOP_PSDUniversal():
        return PSDUniversal(BOS_WORD,BOS_WORD, BOS_WORD, None)

    @staticmethod
    def NULL_PSDUniversal():
        return PSDUniversal(NULL_WORD,NULL_WORD, NULL_WORD, None)

    def construct_by_mrp_node(self, node):
        # herer for verb, or verb prahse, le will be the canoical lemma in PAD vallex_en
        self.le = node.label
        try:
            i = node.properties.index("pos");
            self.pos = node.values[i]
        except:
            logger.error("no pos properties in node:{}".format(json.dumps(node.encode())))
            self.pos = NULL_WORD

        if "frame" in node.properties:
            i = node.properties.index("frame");
            # for psd, only verb has frame
            frame=node.values[i]
            #TODO: cannoical the frame_id into the sense of a lemma
            #worth to mention, the arguments are corresponding the role id in propbank roles.
            # it can be used for future MTL usage, here we hust use the index of this frames, which can be used to recover the frame id
            # Now use we don't predict sense, leave sense as th last mapping steps
            word_id, sense_id = VallexReader.extract_word_and_sense(frame)
            self.sense = sense_id
        else:
            self.sense=NULL_WORD

        if node.anchors:
            self.anchors = node.anchors
        else:
            self.anchors = None


    def construct_by_content(self, pos, le, sense, anchors=None):
        """
        NULL_WORD is empty string, which is just place holder but not break the structure a node
        """
        assert (le == None or isinstance(le, str)) and isinstance(pos, str),(le,pos,sense,anchors)
        self.le = le
        self.pos = pos
        self.sense = sense
        self.anchors = anchors

    def get_frame(self):
        if self.sense == NULL_WORD:
            return NULL_WORD
        else:
            return g_vallex_reader.get_frame_id(self.le, self.sense)

    def __str__(self):
        return self.__repr__()

    def no_anchor_copy(self):
        return PSDUniversal(self.pos, self.le, self.sense, None)

    def to_tuple(self):
        if self.anchors:
            anchors_str = ",".join(["{}-{}".format(x["from"],x["to"]) for x in self.anchors])
        else:
            anchors_str = "N/A"
        return (self.le,self.pos,self.sense, anchors_str)

    def get_anchors_str(self):
        if self.anchors:
            return ",".join(["{}-{}".format(x["from"],x["to"]) for x in self.anchors])
        else:
            return " - "

    def __repr__(self):
        return "({},{},{})-({})".format(self.le, self.pos, self.sense, self.get_anchors_str())

    def __hash__(self):
        return hash(self.__repr__())

    def non_sense_equal(self,other):
        return isinstance(other,PSDUniversal) and self.le == other.le and self.pos == other.pos and self.anchors == other.anchors

    def non_anchors_equal(self,other):
        return isinstance(other,PSDUniversal) and self.le == other.le and self.pos == other.pos and self.sense == other.sense

    def __eq__(self, other):
        return isinstance(other, PSDUniversal) and self.le == other.le and self.pos == other.pos and self.sense == other.sense and self.anchors == other.anchors
