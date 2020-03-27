#!/usr/bin/env python3.6
# coding=utf-8
'''
Parser for DM graph, from both delphi dm or mrp-based dm
@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-19
'''

from collections import defaultdict
from collections import Counter
from nltk.parse import DependencyGraph
from utility.constants import *
from utility.mtool.graph import *
from utility.mtool.codec.sdp import *
import json
import re
import logging

logger = logging.getLogger("mrp.utility.dm_utils")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class DM(DependencyGraph):
    def __init__(self, anno, mrp_graph=None, tokens=None):
        '''
        beside the original dm annotation, or given a mrp graph format, it will also load that into a AMR graph.
        DM has no reentrency, hence variables like that in AMR may not be useful, here we use the id in the nodes as a unique name for each of them.
        '''
        # the realtion triples
        self._triples = []
        # the alignments for the graph
        self._alignments = {}
        # tokens after tokenization and preprocessing
        self._tokens = tokens
        # default value of the nodes is a dict
        self.nodes = defaultdict(lambda: {'address': None,
                                          'type': None,
                                          'head': None,
                                          'rel': None,
                                          'word': None,
                                          'deps': []})
        # Emulate the DependencyGraph (superclass) data structures somewhat.
        # There are some differences, e.g., in AMR it is possible for a node to have
        # multiple dependents with the same relation; so here, 'deps' is simply a list
        # of dependents, not a mapping from relation types to dependents.
        # In typical depenency graphs, 'word' is a word in the sentence
        # and 'address' is its index; here, both point to the object representing
        # the node in DM.

        TOP = Var('TOP')
        # construct top node, also make a top relation to each of the real top nodes in DM
        self.nodes[TOP]['address'] = self.nodes[TOP]['word'] = TOP
        self.nodes[TOP]['type'] = 'TOP'
        if anno:
            # dm , _anno, is a matrix [['A','B','C',],['D','E','F']]
            self._anno = anno
            # parse the original DM matrix into a graph
            self.mrp_graph = matrix2graph(self._anno, framework = "dm", text = None)
            if g is None:
                raise DMForamtError('Well-formedness error in annotation:\n' + matrix2string(self._anno())+"\n")
            self._analyze_mrp_graph(self.mrp_graph)
        elif mrp_graph:
            # load mrp_graph into DMGraph
            # when loading from mrp_graph, there is no more parsimonious info
            # 1. transform every node in mrp_graph.nodes into self.nodes, with its id as variable name
            self._analyze_mrp_graph(self.mrp_graph)
            self._anno = json.dumps(self.mrp_graph.encode())
        else:
            raise NotImplementedError("Both anno(DM Matrix) and mrp_graph is NONE")

    def _analyze_mrp_graph(self, g):
        '''
        Analyze the MRP graph produced by MRP Graph, make it into a specific DM Graph struct
        '''
        # variable to concept mapping
        v2c = {}    # variable -> concept
        # first node should be top
        top_nodes = [node for node in g.nodes if node.is_top]
        # cat training.mrp_dm | grep -oP "\"tops\": \[\d+\]" | grep ","
        # now it seems only one top nodes for each DM, to be verified
        assert len(top_nodes) == 1, "DM {} should have a single top nodes".format(g.id)# ::id bc.cctv_0000.1 ::date 2012-12-11T19:08:03

        # we don't need to walk from top node for DM
        for node in g.nodes:
            v = Var("id"+node.id)
            self.add_node({'address': v, })




        # logger.info("graph: " + str(g.encode()))
        # add top node
        triples = [(Var('TOP'), ':top', v)] + triples
        self._triples = triples

        assert len(self._v2c) == len(g.nodes),  "nodes mismatch {}".format(g.id)
        assert len(self._constants) == total_attributes,  "attributes and constants mismatch {}".format(g.id)
        # top triples = real edges + constants(attributes) + 1 top
        assert len(self._triples) == len(g.edges) + len(self._constants) + 1,  "edges mismatch {}".format(g.id)

        #for v, c in self._v2c.items():
        #    logger.debug("_v2c: v:{}, c :{}".format(v, c))

        #for cons in self._constants:
        #    logger.debug("constants: {}".format(cons))

        #for (h, r, d)  in self._triples:
        #    logger.info("triples: {}, {}, {}".format(h, r, d))

        #for prefix, x in self._index.items():
        #    logger.debug("index: {}, {}".format(prefix, x))


    def _analyze(self, p):

        '''
        Analyze the original matrix
        Make matrix into a DM graph struct, TODO
        '''
        return None

    def triples(
            self,
            head=None,
            rel=None,
            dep=None,
            normalize_inverses=True,
            normalize_mod=True):
        '''
        Returns a list of head-relation-dependent triples in the DM
        Can be filtered by specifying a value (or iterable of allowed values) for:
          - 'head': head variable(s)
          - 'rel': relation label(s) (string(s) starting with ":"), or "core" for all :ARGx roles,
            or "non-core" for all other relations. See also role_triples().
          - 'dep': dependent variable(s)/concept(s)/constant(s)
        Boolean options:
          - 'normalize_inverses': transform (h,':REL-of',d) relations to (d,':REL',h)
            if applicable)
        '''
        # self._triples is the original triples after parsing
        # then do some transformation for each original triple
        tt = (trip for trip in self._triples)
        # if normalize_mod, than always make mod into domain-of
        tt_1 = []
        for h, r, d in tt:
            if normalize_mod and to_be_normalize_mod(r):
                if r == ':domain':
                    x = (h, ":mod-of", d)
                elif r == ":domain-of":
                    x = (h, ":mod", d)
                else:
                    raise NotImplementedError("no normalzied mod is necessary for {}".format(r))
            else:
                x = (h, r, d)

            if normalize_inverses:
                if is_inversed_edge(x[1]):
                    tt_1.append((x[2], get_inversed_edge(x[1]), x[0]))
                else:
                    tt_1.append(x)
            else:
                tt_1.append(x)

        tt = tt_1

        if head:
            tt = (
                (h, r, d) for h, r, d in tt if h in (
                    head if hasattr(
                        head, '__iter__') else (
                        head,)))
        if rel:
            if rel == 'core':
                tt = ((h, r, d) for h, r, d in tt if r.startswith(':ARG'))
            elif rel == 'non-core':
                tt = ((h, r, d) for h, r, d in tt if not r.startswith(':ARG'))
            else:
                tt = (
                    (h, r, d) for h, r, d in tt if r in (
                        rel if hasattr(
                            rel, '__iter__') else (rel)))
        if dep:
            tt = (
                (h, r, d) for h, r, d in tt if d in (
                    dep if hasattr(
                        dep, '__iter__') else (
                        dep,)))
        return list(tt)

    def is_core(self,r):
        return is_core(r)

    def role_triples(self, **kwargs):
        '''
        Same as triples(), but limited to roles (excludes :instance-of, :instance, and :top relations).

        >>> a = AMR('(h / hug-01 :ARG1 (p / person :ARG0-of h))')
        >>> a.role_triples()
        [(Var(h), ':ARG1', Var(p)), (Var(p), ':ARG0-of', Var(h))]
        >>> a.role_triples(head=Var('h'))
        [(Var(h), ':ARG1', Var(p))]
        '''
        tt = [(h, r, d) for h, r, d in self.triples(**kwargs)
              if r not in (':instance', ':instance-of', ':top')]
        return tt

    def constants(self):
        return self._constants

    def concept(self, variable):
        return self._v2c[variable]

    def concepts(self):
        return list(self._v2c.items())

    def var2concept(self):
        return dict(self._v2c)

    def alignments(self):
        return dict(self._alignments)

    def tokens(self):
        return self._tokens

    def __str__(
            self,
            alignments=True,
            tokens=True,
            compressed=False,
            indent=' ' * 4):
        return self._anno


def matrix2string(matrix):
    return '\n'.join(['\t'.join(row) for row in matrix])
