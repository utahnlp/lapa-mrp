#!/usr/bin/env python3.6
# coding=utf-8
'''

EDSReCategorizor use a set of templates built from training corpous and deterministic rules
to recombine/recategorize a fragment of EDS graph into a single node for concept identification.
It also stores frequency of sense for frame concept. (based on training set)
Different from AMR, named entity will not be recategorized,
for now, I only thinks the mwe may can be categorized here.

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-30
'''
from utility.eds_utils.EDSStringCopyRules import *
from utility.dm_utils.SEMIReader import *

from utility.data_helper import *
import logging

logger = logging.getLogger("eds.ReCategorization")

import threading
class EDSReCategorizor(object):

    def entity_templates(self):
        templates = {}
        #templates["name"] = extract_named_entity  # use functions to handle the complexity
        #templates["mwe"] = extract_other_entity
        return templates

    lock = threading.Lock()
    def __init__(self,from_file=False, path="dicts/eds_recategorization",training=False):
        if from_file:
            pass
        else:
            self.ners = {}
            self.wikis = {}
        self.semi_reader = g_semi_reader
        if training == True:
            self.training()
        else:
            self.eval()

    def load_from_txt(self, path="dicts/eds_recategorization.txt"):
        pass


    def save(self, path="dicts/eds_recategorize"):
        """
        Save one pickle and one txt
        """
        pass

    def save_to_txt(self, path="dicts/eds_recategorization.txt",save_counted = False):
        pass

    def unpack_recategorized(self,converted_list,rl,getsense=False,eval= False):
        """
        for mwe, named entity, phrase, we need to some unpack
        if not unpack, we can just all named entity, phrase, mwe as a whole,
        only unpack when creating connected graph
        if unpack here, more concepts will be added for relation identification
        """
        # now we don't do any unpack, hence no any changes for index
        def unpack_one(uni, index):
            def add_concept(uni,dep=0):
                if cat == NULL:
                    return None

                rel_concept.append(uni)
                # here index is the token in dex in the input
                # if not add here, then the token didn't predict any nodes
                indexes.append(index)
                dependent_mark.append(dep)

            # ignore all NULL predication
            if PAD_WORD in [uni.cat] or\
                NULL_WORD in [uni.cat]or\
                UNK_WORD in [uni.cat]:
                return None

            # we hope the concept classifiy can find out all the ners, predict the cat as named
            # TODO: we can do some combination here, to make the relation part better
            # for mwe, the label of each words is the combination of all(joined with “+”)
            # we need to fix them here. if we want to combine into one node
            # For now, we didn't do any combination

            # here fix frame is after combination
            if uni:
                add_concept(uni)
            return None

        rel_concept = []
        indexes = []
        dependent_mark = []
        preentity_id = None
        for i in range(len(converted_list)):
            # for mwe, named enity
            # for pronoun
            uni = converted_list[i]
            preentity_id = unpack_one(uni, i)

        return rel_concept, indexes, dependent_mark

    def acc_list(self,key,dict_list):
        """
        accumlate the count of the key in the dict_list
        dict_list = [(key_i, value)]
        """
        for k_i in dict_list:
            if key == k_i[0]:
                k_i[1] += 1
                return
        dict_list.append([key,1])

    def training(self):
        self.training = True
        logger.info("training mode {}".format(self.training))

    def eval(self,t=None):
        self.training = False

    def convert(self, eds, rl ,snt_token,lemma_token ,pos_token,txt=None):
        return
