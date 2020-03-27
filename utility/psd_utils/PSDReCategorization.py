#!/usr/bin/env python3.6
# coding=utf-8
'''

PSDReCategorizor use a set of templates built from training corpous and deterministic rules
to recombine/recategorize a fragment of PSD graph into a single node for concept identification.
It also stores frequency of sense for frame concept. (based on training set)
Different from AMR, named entity will not be recategorized,
for now, I only thinks the mwe may can be categorized here.

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-30
'''
from utility.psd_utils.PSDGraph import *
from utility.psd_utils.VallexReader import *
from utility.psd_utils.PSDStringCopyRules import *

from utility.data_helper import *
import logging
from nltk.metrics.distance import edit_distance

logger = logging.getLogger("psd.ReCategorization")

import threading

class PSDReCategorizor(object):

    def entity_templates(self):
        templates = {}
        #templates["name"] = extract_named_entity  # use functions to handle the complexity
        #templates["mwe"] = extract_other_entity
        return templates

    lock = threading.Lock()
    def __init__(self,from_file=False, path="dicts/psd_recategorization",training=False):
        if from_file:
            psd_recategorize_f = Pickle_Helper(path)
            psd_recategorize = psd_recategorize_f.load()
            self.senses = psd_recategorize["senses"]
        else:
            self.senses = {}  # record one essential concept in subgraph   c->g could be list
            self.ners = {}
            self.wikis = {}
        # just use the unique global vallex reader
        self.vallexreader = g_vallex_reader
        self.normalize_prob()
        if training == True:
            self.training()
        else:
            self.eval()

    def load_from_txt(self, path="dicts/psd_recategorization.txt"):
        with open(path, 'r') as f:
            json.load(f)


    def save(self, path="dicts/psd_recategorize"):
        """
        Save one pickle and one txt
        """
        psd_recategorizes_f = Pickle_Helper(path)
        psd_recategorizes_f.dump(self.senses, "senses")
        psd_recategorizes_f.save()
        self.save_to_txt(path+".txt")

    def save_to_txt(self, path="dicts/psd_recategorization.txt",save_counted = False):
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
                self.fix_sense_with_uni(uni)
                return uni

            def add_concept(uni,dep=0):
                if len(uni.le) ==0 or uni.le[0] in [",", ".", "(",")","\""] or uni.le[-1] in ["(",")","\""]:
                    return None

                rel_concept.append(uni)
                indexes.append(index)
                dependent_mark.append(dep)

            if PAD_WORD in [uni.pos,uni.le] or\
                NULL_WORD in [uni.pos,uni.le]or\
                UNK_WORD in [uni.pos,uni.le]:
                return None

            uni = try_fix_frame(uni)
            if uni:
                add_concept(uni)
            return None

        rel_concept = []
        indexes = []
        dependent_mark = []
        for i in range(len(converted_list)):
            # for mwe, named enity
            # for pronoun
            uni = converted_list[i]
            preentity_id = unpack_one(uni, i)

        return rel_concept, indexes, dependent_mark

    def fix_sense_with_uni(self, uni):
        """
        only fix sense then the sense is invalid
        """
        uni.sense = self.fix_sense(uni.pos, uni.le, uni.sense)

    def fix_sense(self, pos, le, sense):
        """
        only fix sense then the sense is invalid
        """
        if pos in ['VBD','VB','VBN','VBZ','VBG','VBP']:
            # for verb, we fix it with normlized lemma.
            # we don't trust the sense assigned before relation.
            in_vallex = True
            if le in self.vallexreader.frames:
                if sense not in self.vallexreader.frames[le]:
                    # the sense is not in vallex, fix it with majority
                    in_vallex = False

                # le is in vallex, but sense not correct
                if (not in_vallex) or (sense == NULL_WORD or sense == UNK_WORD or sense is None):
                    # sense is NULL word, then change it
                    sense = self.get_senses(le)
                else:
                    # no fix for other cases
                    pass
            else:
                # if le is not in vallex.
                sense = NULL_WORD
        else:
            # no sense for non root , non verb
            if sense != BOS_WORD:
                sense = NULL_WORD

        return sense

    def get_senses(self, le):
        # for psd, sense may not explictly decided by lemma, and sense
        if not le in self.senses_probs:
            if le in self.vallexreader.frames:
                # here we only choose the first frame for that vallex
                if len(self.vallexreader.frames[le]):
                    return self.vallexreader.frames[le][0]
                else:
                    return "f1"
            else:
                logger.error("le {} is not in vallex, fix lemma not get it correctly".format(le))
                return "f1"
        else:
            probs = self.senses_probs[le] # sense - > role
            return self.most_frequent(probs)[0]  #so far it is worse to use relation feature

    def normalize_prob(self):
        self.senses_probs = {}
        for lemma in self.senses:
            counts = self.senses[lemma]
            self.senses_probs[lemma] = {}
            total = 0 #smoothing
            sense_total = {}
            lemma_freq = self.senses_probs[lemma]
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
                lemma_freq[sen] = prob

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


    def read_senses(self, psd):
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
        out,rel_out,root_index =  psd.node_value(all=True)
        for n, roles in rel_out:   #[[self.graph.node[node]["value"],index], [r,index]]
            le, pos, sen = decompose(n[0])
            # get all the counts for le and pos
            counts =  self.senses.setdefault(le,{})
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

    def convert(self, psd, rl ,snt_token,lemma_token ,pos_token,lemma_str,mwe_token):
        return
