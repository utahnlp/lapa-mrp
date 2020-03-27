#!/usr/bin/env python
#coding=utf-8
'''
Building and hanlding category based dictionary for copying mechanism
Also used by ReCategorization to produce training set, and templates (which partially rely on string matching).
for UCCA, mwe, can be combined

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-28
'''

import threading
from utility.data_helper import *
from utility.ucca_utils.UCCAGraph import *
from utility.constants import *

from nltk.stem.snowball import SnowballStemmer
from nltk.metrics.distance import edit_distance

class UCCARules():
    def __init__(self):
        pass

    #def read_special_joints(self):
    #    with open("dicts/ucca_special_dicts.txt", "r") as fin:
    #        for line in fin:
    #            lemmas = line.splits(":")[0]
    #            expansions = line.splits(":")[1]
    #

    def get_matched_concepts(self,snt_token, ucca_graph, lemma, pos, tok_anchors, ners, mwes):
        """
        here align in ucca_node_value is still anchors in characher position, from, to , here we transform it into aligned token ids.
        return [[n,c,a]]
        """
        results = []
        out = []
        # node_value is an array, [[subnode1, subnode-attr], [subnode2, subnode2_attr]]
        # here subnode-attr is "AMRUniversal" value and "align"
        node_value = ucca_graph.node_value(keys=["value","anchors"])
        # n is node, c is it value UCCAUniversal, a is anchors
        # a here are ahchors[{'from': xx, 'to': }, {'from':, 'to': }]
        for n,c,a in node_value:
            # a here are ahchors[{'from': xx, 'to': }, {'from':, 'to': }]
            # transform anchors into token ids in snt_token array
            # align is [(i, lemma[i], pos[i])]
            align = []
            # usually only one dict in a
            for d in a:
                start = d["from"]
                end = d["to"]
                # tok_anchor is an array of ahchor dict [{'from': xx, 'to': }, {'from':, 'to': }]
                # here tokens may already be combined, then token_anchor will contain more than one dict.
                for i, anchors in enumerate(tok_anchors):
                    # for token i, here may existed more than one original token in it
                    min_start = min([anchor["from"] for anchor in anchors])
                    max_end = max([anchor["to"] for anchor in anchors])
                    if start >= min_start and end <= max_end:
                        align.append(i)
            # adding aligned token index
            ucca_graph[n]['align']=align
            results.append([n,c,align])
        return results
