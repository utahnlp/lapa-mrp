#!/usr/bin/env python3.6
# coding=utf-8
'''

DMReCategorizor use a set of templates built from training corpous and deterministic rules
to recombine/recategorize a fragment of DM graph into a single node for concept identification.
It also stores frequency of sense for frame concept. (based on training set)
Different from AMR, named entity will not be recategorized,
for now, I only thinks the mwe may can be categorized here.

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-30
'''
from utility.dm_utils.dm import *
from utility.dm_utils.DMStringCopyRules import *
from utility.dm_utils.SEMIReader import *

from utility.data_helper import *
import logging

logger = logging.getLogger("dm.ReCategorization")

import threading
class DMReCategorizor(object):

    def entity_templates(self):
        templates = {}
        #templates["name"] = extract_named_entity  # use functions to handle the complexity
        #templates["mwe"] = extract_other_entity
        return templates

    lock = threading.Lock()
    def __init__(self,from_file=False, path="dicts/dm_recategorization",training=False):
        if from_file:
            dm_recategorize_f = Pickle_Helper(path)
            dm_recategorize = dm_recategorize_f.load()
            self.senses = dm_recategorize["senses"]
        else:
            self.senses = {}  # record one essential concept in subgraph   c->g could be list
            self.ners = {}
            self.wikis = {}
        self.normalize_prob()
        self.semi_reader = g_semi_reader
        if training == True:
            self.training()
        else:
            self.eval()

    def load_from_txt(self, path="dicts/dm_recategorization.txt"):
        with open(path, 'r') as f:
            json.load(f)


    def save(self, path="dicts/dm_recategorize"):
        """
        Save one pickle and one txt
        """
        dm_recategorizes_f = Pickle_Helper(path)
        dm_recategorizes_f.dump(self.senses, "senses")
        dm_recategorizes_f.save()
        self.save_to_txt(path+".txt")

    def save_to_txt(self, path="dicts/dm_recategorization.txt",save_counted = False):
        with open(path, 'w+') as data_file:
            json.dump(self.senses, data_file, indent=4)

    def unpack_recategorized(self,converted_list,rl,getsense=False,eval= False):
        """
        for mwe, named entity, phrase, we need to some unpack
        if not unpack, we can just all named entity, phrase, mwe as a whole,
        only unpack when creating connected graph
        if unpack here, more concepts will be added for relation identification
        """
        # now we don't do any unpack, hence no any changes for index
        def unpack_one(uni, index):
            # TODO: fix frame
            def try_fix_frame(uni):
                if getsense and (uni.sense == "" or uni.sense is None):
                    uni.sense = self.get_senses(uni.le, uni.cat)
                return uni

            def add_concept(uni,dep=0):
                if len(uni.le) ==0 or uni.le in [",", ".", "(",")","\""] or uni.le[-1] in ["(",")","\""]:
                    return None

                if getsense and (uni.sense == "" or uni.sense is None):
                    uni.sense = self.get_senses(uni.le, uni.cat)

                rel_concept.append(uni)
                # here index is the token in dex in the input
                # if not add here, then the token didn't predict any nodes
                indexes.append(index)
                dependent_mark.append(dep)

            # ignore all NULL predication
            if (PAD_WORD in [uni.pos,uni.le, uni.cat] or\
                NULL_WORD in [uni.pos,uni.le, uni.cat]or\
                UNK_WORD in [uni.pos,uni.le, uni.cat]) and uni.le not in [':']:
                # exceptions for ':'
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


    def get_senses(self, le, cat):
        # for dm, sense may not explictly decided by lemma, and sense
        if not le in self.senses_probs:
            #TODO: return the most majorty sense
            return "x"
        else:
            if not cat in self.senses_probs[le]:
                # majority sense of that lemma
                max_prob = 0
                max_y = None
                for _,probs in self.senses_probs[le].items():
                    label, prob = self.most_frequent(probs)
                    if prob > max_prob:
                        max_prob = prob
                        max_y = label

                return max_y
            else:
                probs = self.senses_probs[le][cat] # sense - > role
                label, prob = self.most_frequent(probs)
                return label #so far it is worse to use relation feature

    def normalize_prob(self):
        self.senses_probs = {}
        for lemma in self.senses:
            lemma_freqs = self.senses[lemma]
            self.senses_probs[lemma] = {}
            for cat, counts in lemma_freqs.items():
                total = 0 #smoothing
                sense_total = {}
                lemma_cat_freq = self.senses_probs[lemma].setdefault(cat, {})
                for sen,count in counts.items():
                    sense_total[sen] = 5.0
                    total += 5.0
                    for nb in count:
                        total += count[nb]
                        sense_total[sen] += count[nb]

                for sen,count in counts.items():
                    prob = {}
                    prob[None]  = 5.0/sense_total[sen]
                    prob["#prior#"]  = sense_total[sen]/total
                    prob["#total#"]  = total
                    for nb in count:
                        prob[nb] = count[nb]/sense_total[sen]
                    lemma_cat_freq[sen] = prob

    def most_frequent(self,probs):
        #features: [0/1]*len(counts)-1
        max_y = None
        max_p = 0
        for y,prob in probs.items():
            p = prob["#prior#"]
            if p > max_p:
                max_p = p
                max_y = y
        return max_y, max_p


    def read_senses(self, dm):
        """
        count sense,form a dict
        {
         key: lemma,
         value: {
             key2: senInt,
             value2:
                  {
                      key3=negihbor_lemma,
                      value3=IntCount
                  }
          }
        }
        """
        out,rel_out,root_index =  dm.node_value(all=True)
        for n, roles in rel_out:   #[[self.graph.node[node]["value"],index], [r,index]]
            le, pos, cat, sen = decompose(n[0])
            # get all the counts for le and pos
            counts =  self.senses.setdefault(le,{}).setdefault(cat, {})
            # get all counts for sen
            counts[sen] = counts.setdefault(sen,{})

            # use its neighbor lemma as key for the sense counting.
            for r, index in roles:
                nb_le = rel_out[index][0][0].le
                counts[sen][nb_le] = counts[sen].setdefault(nb_le,0)+1

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
        self.normalize_prob()

    def convert(self, dm, rl ,snt_token,lemma_token ,pos_token,txt=None):
        return
