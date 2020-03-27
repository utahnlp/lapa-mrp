#!/usr/bin/env python
#coding=utf-8
'''
Building and hanlding category based dictionary for copying mechanism
Also used by ReCategorization to produce training set, and templates (which partially rely on string matching).
for EDS, mwe, can be combined

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-28
'''

import threading
from utility.data_helper import *
from utility.eds_utils.EDSGraph import *
from utility.constants import *
from utility.dm_utils.SEMIReader import *
from utility.mtool.score.core import *

from nltk.metrics.distance import edit_distance

def normalize_hourofday(string):
    m = {
        'noon': '12',
        'midnight': '0'
        }

    try:
        out = m[string.lower()]
        return out
    except:
        return string

def normalize_dayofweek(string):
    m = {
        'mon': 'Mon',
        'tue': 'Tue',
        'wed': 'Wed',
        'thu': 'Thu',
         'fri': 'Fri',
         'sat': 'Sat',
         'sun': 'Sun'
        }
    s = string.strip()[:3].lower()

    try:
        out = m[s]
        return out
    except:
        raise ValueError('Not a weekday')

def normalize_month(string):
    m = {
        'jan': 'Jan',
        'feb': 'Feb',
        'mar': 'Mar',
        'apr': 'Apr',
         'may': 'May',
         'jun': 'Jun',
         'jul': 'Jul',
         'aug': 'Aug',
         'sep': 'Sep',
         'oct': 'Oct',
         'nov': 'Nov',
         'dec': 'Dec'
        }
    s = string.strip()[:3].lower()

    try:
        out = m[s]
        return out
    except:
        raise ValueError('Not a month')

_float_regexp = re.compile(r"^[-+]?(?:\b[0-9]+(?:\.[0-9]*)?|\.[0-9]+\b)(?:[eE][-+]?[0-9]+\b)?$")
def is_float_re(str):
    return re.match(_float_regexp, str)
super_scripts = '⁰¹²³⁴⁵⁶⁷⁸⁹'
def parseStr(x):
    if x.isdigit():
        if x in super_scripts:
            return super_scripts.find(x)
        return int(x)
    elif is_float_re(x):
        return float(x)
    return None

units = [
"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
"sixteen", "seventeen", "eighteen", "nineteen",
]
tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
scales = ["hundred", "thousand", "million", "billion", "trillion"]
scaless = ["hundreds", "thousands", "millions", "billions", "trillions"]
numwords = {}
numwords["and"] = (1, 0)
for idx, word in enumerate(units):  numwords[word] = (1, idx)
for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)
for idx, word in enumerate(scaless): numwords[word] = (10 ** (idx * 3 or 2), 0)
ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
ordinal_endings = [('ieth', 'y'), ('th', ''), ('st', ''), ('nd', ''), ('rd', '')]


def text2int(textnum):
    if word in ordinal_words:
        return ordinal_words[woru]
    else:
        for ending, replacement in ordinal_endings:
            if word.endswith(ending):
                word = "%s%s" % (word[:-len(ending)], replacement)
        if word not in numwords:
            return None
        return numwords[word][0]

#computing string dissimilarity (e.g. 0 means perfect match)
def disMatch(lemma,con_lemma,t=0.5):
    if lemma == con_lemma: return 0
    # we have a range for EDS align,
    # EDS also have morph changes,
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

class EDSRules():
    """
    rules for EDS, mainly for mwe, ner compunds
    """
    def save(self,filepath="dicts/eds_rule_f"):
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
    def load(self,filepath="dicts/eds_rule_f"):
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

    #def read_special_joints(self):
    #    with open("dicts/eds_special_dicts.txt", "r") as fin:
    #        for line in fin:
    #            lemmas = line.splits(":")[0]
    #            expansions = line.splits(":")[1]
    #

    def clear_freq(self):
        self.lemma_pos_frame_freq = {}
        self.lemmatize_cheat = {}

    def add_lemma_pos_freq(self, snt_tok, tok_lemma,pos_token, uni, freq=1):
        """
        lemma_pos_frame_freq os [pos][old_lemma][eds_con]=freq
        """
        self.lock.acquire()
        pos = pos_token
        word = snt_tok
        le = uni.le
        aux = uni.aux
        cat = uni.cat
        eds_frame = uni.get_label()
        if tok_lemma == le or snt_tok == le:
            freq *= 10
        frame_freq = self.lemma_pos_frame_freq.setdefault(pos,{}).setdefault(word,{})
        frame_freq[eds_frame] = frame_freq.setdefault(eds_frame,0)+freq
        self.lock.release()

    def build_lemma_cheat(self):
        for pos in self.lemma_pos_frame_freq:
            lemma_freqs = self.lemma_pos_frame_freq[pos]
            for lemma in lemma_freqs:
                max_score = 0
                # max_cat should be some majority cat of word
                # TODO: read from the semi
                max_frame = lemma
                for eds_frame in lemma_freqs[lemma]:
                    score =  1.0*lemma_freqs[lemma][eds_frame]
                    assert (score > 0)
                    if score >max_score:
                        max_score = score
                        max_frame = eds_frame

                self.lemmatize_cheat[(lemma,pos)] = max_frame

    #old_ids : batch x (cat,le,lemma,word) only cat is id
    def toEDSSeq(self,pos, snt,lemma, cats, auxs = None,ners = None, mwes = None):
        """
        in this function, use rules to fix the predicated concepts, without chaning the number of nodes
        every node here is still 1-to-1 mapping with the original tokens.
        """
        out = []
        # make sure not delete tokens here
        for i in range(len(snt)):
            aux  = auxs[i] if auxs else None
            ner = ners[i] if ners else None
            mwe = mwes[i] if mwes else None
            txt, le, spos, cat = snt[i], lemma[i], pos[i], cats[i]
            fixed_lemma = self.fix_lemma(i, cat, pos, snt, lemma, mwes)
            if cat and cat != NULL_WORD:
                # for its le,
                # we trust the cat first, but verify it
                cat = self.check_and_fix_cat(txt,fixed_lemma,spos,cat,mwe,ner)
                # for number, named, TODO
                carg = self.fix_carg(txt, fixed_lemma, cat, spos, ner)
                aux = self.get_aux(txt,le,cat)
                uni = EDSUniversal(fixed_lemma,cat,aux,carg,None)
                # fix the uni
                out.append(uni)
            else:
                # copy or determisitc find one
                # if cat is None, we should find one, other wise, we should keep it
                # only fix when tp is not NULL
                if cat == None:
                    cat = self.check_and_fix_cat(txt,fixed_lemma,tp,cat, mwe, ner)

                aux = self.get_aux(txt,le,cat)
                carg = self.fix_carg(txt, fixed_lemma, cat, spos, ner)
                uni = EDSUniversal(fixed_lemma,cat,aux,carg,None)
                out.append(uni)
            #logger.info("i={} is out of bound, snt:{}, lemma:{}, pos:{}, cat:{}".format(i, snt, lemma, pos, cats))
        return out


    def fix_carg(self, token, le, cat, pos, ner):
        """
        60279 "label": "named", "properties": ["carg"]¬
        27459 "label": "card", "properties": ["carg"]¬
        2326 "label": "named_n", "properties": ["carg"]¬
        2215 "label": "mofy", "properties": ["carg"]¬
        2120 "label": "yofc", "properties": ["carg"]¬
        1626 "label": "ord", "properties": ["carg"]¬
        994 "label": "fraction", "properties": ["carg"]¬
        905 "label": "dofw", "properties": ["carg"]¬
        867 "label": "dofm", "properties": ["carg"]¬
        290 "label": "season", "properties": ["carg"]¬
        163 "label": "excl", "properties": ["carg"]¬
        148 "label": "year_range", "properties": ["carg"]¬
        109 "label": "numbered_hour", "properties": ["carg"]¬
        46 "label": "holiday", "properties": ["carg"]¬
        44 "label": "_am_x", "properties": ["carg"]¬
        32 "label": "_pm_x", "properties": ["carg"]¬
        27 "label": "timezone_p", "properties": ["carg"]¬
        5 "label": "meas_np", "properties": ["carg"]¬
        5 "label": "polite", "properties": ["carg"], for please¬
        """
        if cat == NULL_WORD:
            return NULL_WORD
        if cat == 'mofy' and ner == 'DATE':
            # month
            return normalize_month(token)
        elif cat in ["yofc", "dofy"] and ner == "DATE":
            return token
        elif cat == "dofw" and ner == "DATE":
            return normalize_dayofweek(token)
        elif cat == "numbered_hour":
            if re.match(r"\d\d(:\d\d)?", token):
                return token
            else:
                return normalize_hourofday(token)
        elif cat == "season":
            return token.lower()
        elif cat in ["ord", "card", "yofc"]:
            return str(text2int(token))
        elif "named" in cat or cat in ["holiday","timezone_p"]:
            return token
        elif cat == "fraction":
            return token
        elif cat == "excl":
            if token == "--":
                return "s-dash"
            else:
                return token
        elif cat == "_am_x":
            return "am_time"
        elif cat == "_pm_x":
            return "pm_time"
        elif cat.startswith("_"):
            return NULL_WORD
        else:
            return token

    def annotate_mwe(self, data):
        acc = ""
        mwe = ['O']*len(data['lem'])
        start = 0
        skip = False
        i = 0
        for i, le in enumerate(data["lem"]):
            if skip:
                skip = False
            elif len(acc) > 0 and le in self.semi_reader.joints_map.get(acc, []) :
                acc = acc +"+"+le
                if MWE_END in self.semi_reader.joints_map.get(acc,[]):
                    for j in range(start, i+1):
                        mwe[j] = acc
            elif len(acc) > 0 and le == "-" and i < len(data["lem"])-1 \
                and data["lem"][i+1] in self.semi_reader.joints_map.get(acc, []):
                acc = acc +"+"+data["lem"][i+1]
                if MWE_END in self.semi_reader.joints_map.get(acc,[]):
                    for j in range(start, i+2):
                        mwe[j] = acc
                skip = True
            else:
                acc = le
                start = i

        data["mwe"] = mwe
        return data


    def fix_lemma(self, i, cat, pos, snt, stanford_les, mwes=None):
        """
        if a word is not in surface.semi, using its word instead of cannonical lemma
        """
        mwe = mwes[i] if mwes != None else None
        word = snt[i]
        stanford_le = stanford_les[i]
        p = pos[i]
        if mwe != None and mwe!='O':
            if '-' in word:
                return '-+'.join(mwe.split('+'))
            else:
                return mwe
        elif cat in self.semi_reader.abstract_franes:
            return NULL_WORD
        elif stanford_le in self.semi_reader.surface_frames:
            return stanford_le
        elif stanford_le.lower() in self.semi_reader.surface_frames:
            return stanford_le.lower()
        else:
            if stanford_le in ['do', 'be'] and len(stanford_les) > i+1 and stanford_les[i+1] == "not":
                return NULL_WORD
            elif p == 'MD' and len(stanford_les) > i+1 and stanford_les[i+1] == "not":
                return word+"n’t"
            elif stanford_le == "not" and i!= 0:
                return snt[i-1]+"n’t"
            elif word == "a.m.":
                return "am"
            elif word == "p.m":
                return "pm"
            else:
                # for word not in surface form, we still return the word as le as DM,
                # but this lemma should be fixed with cat
                return stanford_le

    def check_and_fix_cat(self, word, le, pos, cat, mwe=None, ner = None):
        """
        make sure the le is already fixed,
        stanford_le, # is pound sign, which is in the semi, but in a cannonical lemma "pound"
        while in eds, # will be used as the lemma
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
                    # cat is not valid, but we still think our model will learn it partialy
                    # fix it
                    tmp_cat = self.get_cat(word, le, pos, cat)
                    if tmp_cat != None:
                        cat = tmp_cat
            else:
                # if not in surface semi, then abstract one, or other special surface frames, just trust it
                # some of the abstract one is not directly the same with training data, such as udef_q become q
                pass
        else:
            # no cat, then try to find one only cat == None, which means we use the copy
            if cat == None:
                cat = self.get_cat(word, le, pos, None)
            else:
                pass

        return cat

    def get_aux(self, word, le, cat):
        if cat == NULL_WORD:
            return NULL_WORD
        if cat in self.semi_reader.abstract_frames:
            return NULL_WORD
        elif le in self.semi_reader.surface_frames:
            semis = self.semi_reader.surface_frames[le]
            auxs = [semi.aux for semi in semis if cat in semi.cat]
            if len(auxs) >0:
                return auxs[0]
            else:
                return ""
        else:
            return "unknown"


    def get_cat(self, word, le, pos, cat = None):
        """
        givem word, le, pos, look up lem first, lemma is the only constraint for semi
        pos is only for reference, aux also may not be correct
        for mwe, targetpos for each token, will be the pos for each token, but the le will be the combination
        we didn't combine in this part, also not in the preprocessing for better MTL.
        we combine in the unpack part.
        """
        # first trust the training data
        if (word, pos) in self.lemmatize_cheat:
            frame = self.lemmatize_cheat[(word, pos) ]
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

    def get_matched_concepts(self,ori_text, snt_token, eds_graph, lemma, pos, mwe_token, tok_anchors):
        """
        here align in eds_node_value is still anchors in characher position, from, to , here we transform it into aligned token ids.
        return [[n,c,a]]
        """
        results = []
        # node_value is an array, [[subnode1, subnode-attr], [subnode2, subnode2_attr]]
        node_value = eds_graph.node_value(keys=["value","anchors"])
        # here subnode-attr is "AMRUniversal" value and "align"
        used_align = {}
        special_nodes = {}
        # n is node, c is it value EDSUniversal, a is anchors
        # a here are ahchors[{'from': xx, 'to': }, {'from':, 'to': }]
        for n,c,a in node_value:
            # a here are ahchors[{'from': xx, 'to': }, {'from':, 'to': }]
            # transform anchors into token ids in snt_token array
            # align is [(i, lemma[i], pos[i])]
            align = []
            containing_align = []
            # usually only one dict in a
            for d in a:
                # it is useful for DM and EDS, but not for EDS
                start = d["from"]
                end = d["to"]
                while start < end and ori_text[start] in PUNCTUATION:
                    start += 1;
                while end > start and ori_text[end - 1] in PUNCTUATION:
                    end -= 1;
                # tok_anchor is an array of ahchor dict [{'from': xx, 'to': }, {'from':, 'to': }]
                # here tokens may already be combined, then token_anchor will contain more than one dict.
                for i, anchors in enumerate(tok_anchors):
                    # for token i, here may existed more than one original token in it
                    min_start = min([anchor["from"] for anchor in anchors])
                    max_end = max([anchor["to"] for anchor in anchors])
                    if start >= min_start and end <= max_end:
                        #if c.cat not in ['udef_q','proper_q','pronoun_q', 'def_explicit_q', 'of_p', 'def_implicit_q','compound', 'appos', 'nominalization']:
                        if c.cat not in ['compound', 'appos']:
                            # it is a regular termninal node
                            align.append(i)
                            used_align.setdefault(i, []).append(n)
                        else:
                            containing_align.append(i)
                    elif min_start >= start and max_end <= end:
                        containing_align.append(i)

            # here change the character anchors to tokens, we need rules to get most of them aligned.
            # adding aligned token index
            # here when no exact aligning, the align will be empty. Now just leave it empty, leave it for latent to learn
            if len(align) > 0:
                eds_graph[n]['align']=align
            else:
                # check the alignment within the given tokens
                filtered_unalign = []
                for j in containing_align:
                    if j not in used_align:
                        filtered_unalign.append(j)

                further_align = self.match_concept(snt_token, c, lemma, pos, mwe_token,filtered_unalign)
                if len(further_align)> 0:
                    eds_graph[n]['align'] = further_align
                    used_align.setdefault(i,[]).append(n)
                else:
                    # when this abstract ndoe cannot be aligned, it is must be a special non-terminial node, add its label tag to tags of its containing align
                    eds_graph[n]['align'] = []
                    eds_graph[n]['combined_align'] = containing_align
                    special_nodes.setdefault(n, []).extend(containing_align)

        for n, containing_align in special_nodes.items():
            cat = eds_graph[n]["value"].cat
            for i, index in enumerate(containing_align):
                if i == 0:
                    # first node
                    tag = "B_" + cat
                else:
                    tag = "I_" + cat

                if index not in used_align:
                    # the token is not used yet, adding a new token
                    eds_graph.add_companion_tag_node(n, cat, tag, index)
                else:
                    # if it is a original node, we just set the tag for it
                    if len(used_align[index]) > 1:
                        logger.error("{}, '{}' is aligned to multiple nodes{}".format(index,snt_token[index], [eds_graph[x]["value"] for x in used_align[index]]))

                    eds_graph[used_align[index][0]]["value"].tag = eds_graph[used_align[index][0]]["value"].tag + "|" + tag

        node_value = eds_graph.node_value(keys=["value","align"])
        # TODO: check the cat, if the cat is deterministic from the surface part, we make it for copy
        # we hope this can help to remove those phrase, long and sparse cats
        for n,c,a in node_value:
            can_copy = False
            if c.le in self.semi_reader.surface_frames:
                semis = self.semi_reader.surface_frames[c.le]
                cats = set([s.cat for s in semis])
                if len(cats) == 1 and c.cat in cats:
                    # exactly match
                    can_copy = True
                else:
                    can_copy = False
            elif c.cat.endswith("_unknown"):
                can_copy = True

            # mark the rest of token tag as O for special nodes
            if c.tag == None:
                c.tag = 'O'

            results.append([n,c,a,can_copy])

        return results


    def match_concept(self,snt_token,concept,lemma,pos,mwe_token, candidate = None):
        """
        use distant match to find possible alignment first.
        """
        le,cat,aux,carg,tag = decompose(concept)
        target_le = carg if carg else le
        align = []
        if candidate is None:
            candidate = range(len(snt_token))
        for i in candidate:
            # we need it less than 0.5
            if lemma[i] in target_le:
                align.append(i)
                continue
            elif disMatch(lemma[i],target_le) < 1:
                align.append(i)
                continue

        return align
