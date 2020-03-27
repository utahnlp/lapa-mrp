#!/usr/bin/env python
#coding=utf-8
'''
Building and hanlding category based dictionary for copying mechanism
Also used by ReCategorization to produce training set, and templates (which partially rely on string matching).
for DM, mwe, can be combined

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-28
'''

import threading
from utility.data_helper import *
from utility.dm_utils.DMGraph import *
from utility.constants import *
from utility.dm_utils.SEMIReader import *

from nltk.stem.snowball import SnowballStemmer
from nltk.metrics.distance import edit_distance

#computing string dissimilarity (e.g. 0 means perfect match)
def semi_disMatch(lemma,con_lemma,t=0.5):
    if lemma == con_lemma: return 0
    if (con_lemma in lemma  or  lemma in  con_lemma) and len(lemma)>2 and len(con_lemma)>2 :
        return 0
    if lemma.endswith("ily") and lemma[:-3]+"y"==con_lemma:
        return 0
    if lemma.endswith("ing") and (lemma[:-3]+"e"==con_lemma or lemma[:-3]==con_lemma):
        return 0
    if lemma.endswith("ical") and lemma[:-4]+"y"==con_lemma:
        return 0
    if lemma.endswith("ially") and lemma[:-5] in con_lemma:
        return 0
    if lemma.endswith("ion") and (lemma[:-3]+"e"==con_lemma or lemma[:-3]==con_lemma):
        return 0
    if lemma in con_lemma and len(lemma)>3 and len(con_lemma)-len(lemma)<5:
        return 0
    if lemma.endswith("y") and lemma[:-1]+"ize"==con_lemma:
        return 0
    if lemma.endswith("er") and (lemma[:-2]==con_lemma or lemma[:-3]==con_lemma  or lemma[:-1]==con_lemma):
        return 0
    dis = 1.0*edit_distance(lemma,con_lemma)/min(12,max(len(lemma),len(con_lemma)))
    if (dis < t ) :
        return dis
    return 1


class DMRules():
    """
    rules for DM, mainly for mwe, ner compunds
    """
    def save(self,filepath="dicts/dm_rule_f"):
        pickle_helper= Pickle_Helper(filepath)
        pickle_helper.dump([k for k in self.lemma_pos_frame_freq.keys()],"keys")
        for pos in self.lemma_pos_frame_freq:
            pickle_helper.dump(self.lemma_pos_frame_freq[pos], pos)
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
    def load(self,filepath="dicts/dm_rule_f"):
        pickle_helper= Pickle_Helper(filepath)
        data = pickle_helper.load()
        keys = data["keys"]
        self.lemma_pos_frame_freq = {}
        for key in keys:
            self.lemma_pos_frame_freq[key] = data[key]
        self.build_lemma_cheat()
        return self

    def set_rules(self):
        """
        set amr rules for each category, the valis is a function
        """
        self.rules = {}
        # now didn't use amu rules.

    def __init__(self):
        # lemma_pos_frame_freq os [cat][old_lemma][amr_con]=freq
        self.lemma_pos_frame_freq = {}
        # lemmatize_cheat => [cat, lemma] -> best_amr_lemma
        self.semi_reader = g_semi_reader
        self.lemmatize_cheat = {}
        self.special_joints_map = {}
        # self.read_special_joints()
        # when initial, build_lemma_cheat will only consider the propbank cat and verbalization  cats, cat is Rule_Predicate
        self.build_lemma_cheat()
        # add rules function
        self.set_rules()
        self.stemmer = SnowballStemmer("english")

    #def read_special_joints(self):
    #    with open("dicts/dm_special_dicts.txt", "r") as fin:
    #        for line in fin:
    #            lemmas = line.splits(":")[0]
    #            expansions = line.splits(":")[1]
    #

    def clear_freq(self):
        self.lemma_pos_frame_freq = {}
        self.lemmatize_cheat = {}

    def add_lemma_pos_freq(self, snt_tok, tok_lemma,uni, freq=1):
        """
        lemma_pos_frame_freq os [pos][old_lemma][amr_con]=freq
        """
        self.lock.acquire()
        pos = uni.pos
        le = uni.le
        sense = uni.sense
        cat = uni.cat
        dm_frame = uni.get_frame()
        if tok_lemma == le or snt_tok == le:
            freq *= 10
        frame_freq = self.lemma_pos_frame_freq.setdefault(pos,{}).setdefault(le,{})
        frame_freq[dm_frame] = frame_freq.setdefault(dm_frame,0)+freq
        self.lock.release()

    def build_lemma_cheat(self):
        for pos in self.lemma_pos_frame_freq:
            lemma_freqs = self.lemma_pos_frame_freq[pos]
            for lemma in lemma_freqs:
                max_score = 0
                # max_cat should be some majority cat of word
                # TODO: read from the semi
                max_frame = lemma
                for dm_frame in lemma_freqs[lemma]:
                    score =  1.0*lemma_freqs[lemma][dm_frame]
                    assert (score > 0)
                    if score >max_score:
                        max_score = score
                        max_frame = dm_frame

                self.lemmatize_cheat[(lemma,pos)] = max_frame

    def check_cat_valid_in_lemma_cheat(self, pos, lemma, cat):
        if pos in self.lemma_pos_frame_freq:
            if lemma in self.lemma_pos_frame_freq[pos]:
                for frame in self.lemma_pos_frame_freq[pos][lemma]:
                    if ':' in frame:
                        if cat == frame.split(':')[0]:
                            return True
                        else:
                            pass
                    else:
                        if cat == frame:
                            return True
                return False
            else:
                return False
        else:
            return False

    def check_lemma_existed_in_lemma_cheat(self, pos, lemma):
        if pos in self.lemma_pos_frame_freq:
            if lemma in self.lemma_pos_frame_freq[pos]:
                return True
            else:
                return False
        else:
            return False

    #old_ids : batch x (cat,le,lemma,word) only cat is id
    def toDMSeq(self,pos, snt,lemma, pred_lemma, cats, senses = None,ners = None, mwes = None):
        """
        in this function, use rules to fix the predicated concepts, without chaning the number of nodes
        every node here is still 1-to-1 mapping with the original tokens.
        pred_lemma can be None, when no lemma classification used
        """
        out = []
        # make sure not delete tokens here
        for i in range(len(snt)):
            sense  = senses[i] if senses else None
            ner = ners[i] if ners else None
            mwe = mwes[i] if mwes else None
            txt, le, tp, cat = snt[i], lemma[i], pos[i], cats[i]
            if pred_lemma == None or pred_lemma[i] == None:
                fixed_lemma = self.fix_lemma(out, i, pos, snt, lemma, ners, cats, mwes)
            else:
                fixed_lemma = pred_lemma[i]
            if cat and cat != NULL_WORD:
                # for its le,
                # we trust the cat first, but verify it
                cat = self.check_and_fix_cat(txt,fixed_lemma,tp,cat,mwe,ner)
                uni = DMUniversal(tp,cat,sense,fixed_lemma,None)
                # fix the uni
                out.append(uni)
            else:
                # copy or determisitc find one
                # if cat is None, we should find one, other wise, we should keep it
                # only fix when tp is not NULL
                if tp != NULL_WORD:
                    cat = self.check_and_fix_cat(txt,fixed_lemma,tp,cat, mwe, ner)

                uni = DMUniversal(tp,cat,sense,fixed_lemma,None)
                out.append(uni)
            #logger.info("i={} is out of bound, snt:{}, lemma:{}, pos:{}, cat:{}".format(i, snt, lemma, pos, cats))
        return out

    def fix_lemma(self, out, i, pos, snt, stanford_les, ners, cats, mwes = None):
        """
        if a word is not in surface.semi, using its word instead of cannonical lemma
        the order of the pipelienis important
        """
        mwe = mwes[i] if mwes != None else None
        word = snt[i]
        stanford_le = stanford_les[i]
        tp = pos[i]
        high_pos = self.semi_reader.get_high_pos(tp)
        if stanford_le.lower() in ['do', 'have','will'] and len(stanford_les) > i+1 and stanford_les[i+1].lower() == "not":
            return NULL_WORD
        elif (tp == 'MD' or word.lower() in ['can','could','would','is','am','are','was','were']) and len(stanford_les) > i+1 and stanford_les[i+1] == "not":
            return word+"n’t"
        elif stanford_le.lower() == "not" and i!= 0 and (pos[i-1] == 'MD' or stanford_les[i-1] in ['can','could','do','have','would','be','will']) :
            return snt[i-1].lower()+"n’t"
        elif self.check_mwe(i, pos, mwes, out):
            # pass the not rule first
            if '-' in word:
                return '-+'.join(mwe.split('+'))
            else:
                return mwe
        elif self.check_word_valid(word, ners[i], tp, high_pos, cats[i]):
            return word.lower()
        elif stanford_le.lower() in self.semi_reader.surface_frames:
            if stanford_le.lower() == "be" and word.lower() in ['’s','is','am','are','was','were','been']:
                return word.lower()
            elif word.lower() in ['an']:
                return word.lower()
            else:
                return stanford_le.lower()
        elif stanford_le == '#':
            return "pound"
        else:
            if high_pos in ['v','a','n']:
                possible_lemmas = []
                test_word = word
                if '-' in word:
                    # but it is not mwe, use the last part of it
                    test_word = word.split('-')[-1]

                if test_word.endswith("ing"):
                    possible_lemmas.append(test_word[:-3])
                    possible_lemmas.append(test_word[:-3]+"e")
                elif test_word.endswith("ed"):
                    possible_lemmas.append(test_word[:-2])
                    possible_lemmas.append(test_word[:-2]+"e")

                possible_lemmas.append(self.stemmer.stem(test_word))

                for x in possible_lemmas:
                    if x in self.semi_reader.surface_frames:
                        return x
                return test_word.lower()
            else:
                return word.lower()

    def check_word_valid(self, word, ner, tpos, high_pos, cat):
        """
        check whether using words as lemma is correct
        """
        if ner != 'O' and len(word)>0 and word[0].isupper():
            return True
        if word in self.semi_reader.surface_frames:
            semis = self.semi_reader.surface_frames[word]
            cats = [semi.cat for semi in semis]
            if any([cat.startswith(high_pos) for cat in cats]):
                if self.check_cat_valid_in_lemma_cheat(tpos, word, cat):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
            #if self.check_cat_valid_in_lemma_cheat(tpos, word, cat):
            #    return True
            #else:
            #    return False

    def find_lemma_in_frames(self, le):
        # TODO: here we trust the mwe, may it is not good
        # we need make those not easily decide answer to classify
        best_dis = .4
        best_lemma = NULL_WORD
        for con_lemma in self.semi_reader.surface_frames:
            dis = semi_disMatch(le,con_lemma)
            if dis < best_dis :
                best_dis = dis
                best_lemma = con_lemma
        if best_lemma != NULL_WORD:
            return best_lemma
        else:
            return le

    def check_mwe(self, i, tpos, mwes, out):
        if mwes != None and mwes[i] !='O':
            # do mwe only the previous mwe word use that as lemma
            if i >  0 and mwes[i-1] != 'O':
                if out[i-1].le.lower() == mwes[i-1].lower():
                    return True
                else:
                    return False
            else:
                if not self.check_lemma_existed_in_lemma_cheat(tpos[i], mwes[i]):
                    # unseen one will droFalseped
                    return False
                else:
                    return True
        else:
            return False


    def check_and_fix_cat(self, word, le, tpos, cat, mwe=None, ner = None):
        """
        make sure the le is already fixed,
        stanford_le, # is pound sign, which is in the semi, but in a cannonical lemma "pound"
        while in dm, # will be used as the lemma
        given word, fixed_lemma, fix the pos
        """
        if cat not in [None, NULL_WORD]:
            # we have a cat, try verfiy it
            # here we need an extra mapping for special tokens to lookup
            if cat in self.semi_reader.abstract_frames:
                # if it is abstract, don't change it
                pass
            elif le in self.semi_reader.surface_frames:
                # then cat should be in the semi_reader cat
                semis = self.semi_reader.surface_frames[le]
                cats = [semi.cat for semi in semis]
                if cat in cats:
                    # cat is valid
                    pass
                else:
                    # cat is not valid or when cat in not in semi, use lemmatize_cheat
                    # fix it
                    if self.check_cat_valid_in_lemma_cheat(tpos, le, cat):
                        pass
                    else:
                        tmp_cat = self.get_cat(word, le, tpos, cat)
                        if tmp_cat != None:
                            cat = tmp_cat
            else:
                # if not in surface semi, then abstract one, or other special surface frames, just trust it
                # some of the abstract one is not directly the same with training data, such as udef_q become q
                pass
        else:
            # no cat, then try to find one only cat == None, which means we use the copy
            if cat == None:
                cat = self.get_cat(word, le, tpos, None)
            else:
                pass

        return cat

    def get_cat(self, word, le, tpos, cat = None):
        """
        givem word, le, tpos, look up lem first, lemma is the only constraint for semi
        pos is only for reference, sense also may not be correct
        for mwe, targetpos for each token, will be the pos for each token, but the le will be the combination
        we didn't combine in this part, also not in the preprocessing for better MTL.
        we combine in the unpack part.
        """
        # first trust the training data
        if (le, tpos) in self.lemmatize_cheat:
            frame = self.lemmatize_cheat[(le, tpos) ]
            cat = frame.split(':')[0]
        else:
            # if not in lemmatize_cheat
            # lookup semi dict
            if le in self.semi_reader.surface_frames:
                semis = self.semi_reader.surface_frames[le]
                cats = [semi.cat for semi in semis]
                if len(semis) == 1:
                    cat = semis[0].cat
                else:
                    # using pos to help selection
                    # semi_pos = self.semi_reader.get_pos(tpos)
                    # let just try the most similar cat with predicted cat
                    # when code get there, it means cat must be not valid
                    if cat != None:
                        if cat in ['v','a','n','p','c','q','x']:
                            high_pos = cat
                        elif cat.split("_")[0] in ['v','a','n','p','c','q','x']:
                            high_pos = cat.split("_")[0]
                        else:
                            high_pos = self.semi_reader.get_high_pos(tpos)
                    else:
                        high_pos = self.semi_reader.get_high_pos(tpos)

                    cats_with_high_pos = [x for x in cats if x.startswith(high_pos)]
                    if len(cats_with_high_pos) > 0:
                        cat = cats_with_high_pos[0]
                    else:
                        cats_with_v = [x for x in cats if x.startswith("v")]
                        if len(cats_with_v) > 0:
                            cat = cats_with_v[0]
                        else:
                            cat = cats[0]
            else:
                # cat is None, and le is not in semi, than, the le must be wrong
                if cat == None:
                    cat = self.semi_reader.get_high_pos(tpos)

        return cat

    def get_matched_concepts(self,snt_token, dm_graph, lemma, pos, tok_anchors, ners, mwes):
        """
        here align in dm_node_value is still anchors in characher position, from, to , here we transform it into aligned token ids.
        return [[n,c,a]]
        """
        results = []
        out = []
        # node_value is an array, [[subnode1, subnode-attr], [subnode2, subnode2_attr]]
        # here subnode-attr is "AMRUniversal" value and "align"
        node_value = dm_graph.node_value(keys=["value","anchors"])
        # n is node, c is it value DMUniversal, a is anchors
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
            dm_graph[n]['align']=align
            # TODO: check the cat, if the cat is deterministic from the surface part, we make it for copy
            # we hope this can help to remove those phrase, long and sparse cats
            can_copy = False
            if c.le in self.semi_reader.surface_frames:
                semis = self.semi_reader.surface_frames[c.le]
                cats = set([s.cat for s in semis])
                if c.cat in cats:
                    if len(cats) == 1:
                        # exactly match
                        can_copy = True
                    else:
                        pass
                else:
                    # some cat are not in semi
                    pass
            can_le_copy = False
            if len(align) == 1:
                rule_lemma = self.fix_lemma(out, align[0], pos, snt_token, lemma, ners, mwes)
                if rule_lemma.lower() == c.le.lower():
                    can_le_copy = True
                else:
                    # not handl multple aligns
                    pass

            out.append(c)
            results.append([n,c,align,can_copy, can_le_copy])
        return results
