#!/usr/bin/env python
#coding=utf-8
'''
Building and hanlding category based dictionary for copying mechanism
Also used by ReCategorization to produce training set, and templates (which partially rely on string matching).
for PSD, mwe, can be combined

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-28
'''

import threading
from utility.data_helper import *
from utility.psd_utils.PSDGraph import *
from utility.psd_utils.VallexReader import *
from utility.constants import *
import re

from nltk.metrics.distance import edit_distance

#computing string dissimilarity (e.g. 0 means perfect match)
def vallex_disMatch(lemma,con_lemma,t=0.5):
    if lemma == con_lemma: return 0
    #if (con_lemma in lemma or lemma in  con_lemma) and len(lemma)>2 and len(con_lemma)>2 :
    #    return 0
    dis = 1.0*edit_distance(lemma,con_lemma)/min(12,max(len(lemma),len(con_lemma)))
    if (dis < t ) :
        return dis
    return 1

class PSDRules():
    """
    rules for PSD, mainly for mwe, ner compunds
    """
    def save(self,filepath="dicts/psd_rule_f"):
        pickle_helper= Pickle_Helper(filepath)
        pickle_helper.dump(self.lemma_freq, "lemma_freq_dicts")
        pickle_helper.save()

        self.load(filepath)

    @staticmethod
    def unmixe(mixed,threshold = 5):
        # use high frequency and low frequency for other non-const nodes
        high_frequency = dict()
        low_frequency = dict()
        for i in mixed:
            # No text normalization found in psd
            if mixed[i][0] >= threshold:
                high_frequency[i] = mixed[i]
            else:
                low_frequency[i] = mixed[i]
        return high_frequency,low_frequency

    lock = threading.Lock()
    def load(self,filepath="dicts/psd_rule_f"):
        pickle_helper= Pickle_Helper(filepath)
        data = pickle_helper.load()
        self.lemma_freq = data["lemma_freq_dicts"]
        self.build_lemma_cheat()
        return self

    def set_rules(self):
        """
        set amr rules for each category, the valis is a function
        """
        self.rules = {}
        # now didn't use amu rules.

    def __init__(self):
        self.lemma_freq = {}
        # lemmatize_cheat => [cat, lemma] -> best_amr_lemma
        self.lemmatize_cheat = {}
        self.vallex_reader = g_vallex_reader
        # when initial, build_lemma_cheat will only consider the propbank predicate and verbalization  predicates, cat is Rule_Predicate
        self.build_lemma_cheat()
        # add rules function
        self.set_rules()

    def clear_freq(self):
        self.lemma_freq = {}
        self.lemmatize_cheat = {}

    def add_lemma_freq(self, tok_lemma, psd_lemma, freq=1,sense=NULL_WORD):
        """
        lemma_freq os [old_lemma][amr_con]=freq
        """
        self.lock.acquire()
        lemma_freq = self.lemma_freq.setdefault(tok_lemma,{})
        lemma_freq[psd_lemma] = lemma_freq.setdefault(psd_lemma,0)+freq
        self.lock.release()

    def build_lemma_cheat(self):
        # only verb has frame, pos is not useful here.
        for lemma in self.lemma_freq:
            max_lemma = lemma
            max_score = 0.0
            for psd_lemma in self.lemma_freq[lemma]:
                score =  1.0*self.lemma_freq[lemma][psd_lemma]
                assert (score > 0)
                if score >max_score:
                    max_score = score
                    max_lemma = psd_lemma

            # here lemma is token lemma
            self.lemmatize_cheat[lemma] = max_lemma

    def toPSDSeq(self,pos, snt,lemma, high, senses = None,ners = None, mwes = None):
        """
        Pay attention, during this, we should not delete any nodes, other wise, the anchors will be wrong,
        """
        out = []
        for i in range(len(snt)):
            sense  = senses[i] if senses else None
            mwe = mwes[i] if mwes else None
            txt, le, tp, h = snt[i], lemma[i], pos[i], high[i]
            if h and h != NULL_WORD:
                h = self.check_and_fix_lemma(tp,txt,le, h, mwe)
                uni = PSDUniversal(tp,h,sense,None)
                out.append(uni)
            else:
                # h is None, means it is for copy, we need some high precision rules here.
                if h == None:
                    h = self.check_and_fix_lemma(tp, txt, le, h, mwe)
                    uni = PSDUniversal(tp,h,sense,None)
                    out.append(uni)
                else:
                    uni = PSDUniversal(tp,h,sense,None)
                    out.append(uni)

        return out


    def check_and_fix_lemma(self, tpos, tok, stanford_le, psd_le, mwe):
        """
        lemma must be fixed for PSD, because here the le we take is either from the high-dict or from the the lemma
        we must use the lemmatize_cheat or frames to fix it.
        """
        if psd_le not in [None, NULL_WORD]:
            # if psd_le is predicted, check whether it is valid
            if tpos in ['VBD','VB','VBN','VBZ','VBG','VBP']:
                # then psd_le should in vallex_reader frames
                if psd_le not in self.vallex_reader.frame_lemmas:
                    lemma = self.fix_lemma(tpos,tok, stanford_le, psd_le, mwe)
                else:
                    lemma = psd_le
            elif tpos in ['CD']:
                lemma = self.try_number_lemma(tpos,tok, stanford_le)
            else:
                lemma = psd_le
        else:
            if psd_le == None:
                lemma = self.fix_lemma(tpos,tok, stanford_le, psd_le, mwe)
            else:
                # predicted as NULL WORD, not fix it
                lemma = psd_le
        return lemma

    def fix_lemma(self, tpos,tok, stanford_le, psd_le, mwe):
        if tpos in ['CD']:
            lemma = self.try_number_lemma(tpos, tok, stanford_le)
        elif tpos in ['VBD','VB','VBN','VBZ','VBG','VBP']:
            lemma = self.find_lemma_in_vallex(stanford_le,mwe)
        else:
            lemma = stanford_le

        return lemma

    def find_lemma_in_vallex(self, le, mwe):
        # TODO: here we trust the mwe, may it is not good
        # we need make those not easily decide answer to classify
        # this is very slow
        if mwe != None and mwe != 'O': return mwe
        best_dis = .4
        best_lemma = NULL_WORD
        for con_lemma in self.vallex_reader.frame_lemmas:
            dis = vallex_disMatch(le,con_lemma)
            if dis < best_dis :
                best_dis = dis
                best_lemma = con_lemma
                if best_dis == 0:
                    return best_lemma

        if best_lemma != NULL_WORD:
            return best_lemma
        else:
            return le

    def try_number_lemma(self, pos,tok, lemma):
        if pos == "CD":
            if re.match(r'\d\d+s', tok):
                # 1930s
                return tok
            else:
                # float, the lemma is itself
                return re.sub("[,\/]","_", lemma)
        else:
            return lemma

    def get_matched_concepts(self,snt_token, psd_graph, lemma, pos, mwe_token, tok_anchors):
        """
        here align in psd_node_value is still anchors in characher position, from, to , here we transform it into aligned token ids.
        return [[n,c,a]]
        """
        results = []
        # node_value is an array, [[subnode1, subnode-attr], [subnode2, subnode2_attr]]
        # here subnode-attr is "AMRUniversal" value and "align"
        node_value = psd_graph.node_value(keys=["value","anchors"])
        # n is node, c is it value PSDUniversal, a is anchors
        # a here are ahchors[{'from': xx, 'to': }, {'from':, 'to': }]
        for n,c,a in node_value:
            # a here are ahchors[{'from': xx, 'to': }, {'from':, 'to': }]
            # transform anchors into token ids in snt_token array
            # align is [(i, lemma[i], pos[i])]
            align = []
            token = ""
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
            token_str = " ".join([snt_token[i] for i in align])
            # in PSD, bring in, will become "bring_in", but only the anchors for bring will be used.
            # hence, when a lemma is in the label we still treat it is copy from lemma
            lemma_str = " ".join([lemma[i] for i in align])
            # adding aligned token index
            psd_graph[n]['align']=align
            can_copy_mwe = c.le in [ mwe_token[x] for x in align if mwe_token[x] != 'O']
            # use token, use le or phrase, use lowercase token
            can_be_copied = (token_str.lower() == c.le.lower() or token_str.lower() == c.le.lower() or lemma_str in c.le or self.try_number_lemma(c.pos, token_str, lemma_str) == c.le)
            # for copied, we should handle all the rest cases by rules.
            results.append([n,c,align, can_be_copied or can_copy_mwe])
        return results
