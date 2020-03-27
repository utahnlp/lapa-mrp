#!/usr/bin/env python3.6
# coding=utf-8
'''

This reader reads all amr propbank file,
and add possible cannonical amr lemma
to the corresponding copying dictionary of a word and aliases of the word

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-28
'''

from utility.constants import *
from utility.data_helper import folder_to_files_path
import re
import logging

logger = logging.getLogger("mrp")

arg_reg=r"(?P<role>\S+)\s(?P<arg>\w+)(?P<attach>\s\{.*?\})?,?"

def add_concept(lemmas_to_concept,le,con):

    if not le in lemmas_to_concept:
        lemmas_to_concept[le]= set([con])
    else:
        lemmas_to_concept[le].add(con)

class SEMI():
    def __init__(self, lemma, cat, aux, args, arg_sign, dm_frame, eds_label):
        self.lemma = lemma
        self.cat = cat
        self.aux = aux
        self.args = args
        self.arg_sign = arg_sign
        self.dm_frame = dm_frame
        self.eds_label = eds_label

class SEMIReader:
    # http://moin.delph-in.net/ErgLeTypes, are the pos types.
    # some of them are merged, v, n,a, p, c, q, x other

    def get_high_pos(self, pos):
        if pos in ['VBD','VBG','VBN','VBP','VB','VBZ']:
            #NN, JJ if there is no n or a in semi, it use v
            return 'v'
        elif pos in ['JJ','JJR',"JJS",'RB','RBR','RBS','RP']:
            return "a"
        elif pos in ['NN','NNS','NNP','NNPS']:
            return "n"
        elif pos in ['IN','TO']:
            return "p"
        elif pos in ['CC']:
            return "c"
        elif pos in ['DT','PDT','PRP$','WRB','WP','PRP']:
            return "q"
        else:
            return "x"

    def parse(self):
        """
        parse all the propbank frames
        """
        self.surface_frames = dict()
        self.abstract_frames = dict()
        self.joints = set()
        self.joints_map = {}
        self.parse_semi_file(self.surface_smi_path)
        self.parse_semi_file(self.abstract_smi_path)

    def __init__(self, folder_path=semi_folder_path):
        self.surface_smi_path = folder_path + "surface.smi"
        self.abstract_smi_path = folder_path + "abstract.smi"
        # TODO: hierarchical smi ?
        self.hierarchy_smi_path = folder_path + "hierarchy.smi"
        self.parse()

    @staticmethod
    def get_arg_list_from_semi_a(semi_a):
        semi_a = semi_a.lstrip().rstrip()
        #_abaculus_n_1 : ARG0 x { IND + }.
        # delete last .
        x = semi_a[:-1].lstrip().rstrip()
        args = re.findall(arg_reg, x)
        return args

    @staticmethod
    def get_lem_cat_aux_from_semi_p(semi_p):
        semi_p = semi_p.lstrip().rstrip()
        if semi_p.startswith("_"):
            ss = semi_p.split("_")
            # * 19104 _1
            # * 5828    no aux
            # * 45 _2
            # here lemma, we only consider the first part
            lemma = ss[1]
            # unknown is involced in cat
            if ss[-1] in ["1", "2", "unknown"]:
                cat = "_".join(ss[2:-1])
                aux = ss[-1]
            else:
                cat = "_".join(ss[2:])
                aux = ""
            return lemma, cat, aux
        else:
            return "",semi_p,""

    @staticmethod
    def get_dm_frame(cat, args):
        arg_signatures = SEMIReader.get_arg_signature(args)
        try:
            return cat +":"+arg_signatures
        except:
            logger.error("%s %s", type(cat), type(arg_signatures))

    @staticmethod
    def get_eds_label(lemma, cat, aux):
        """
        lemma is the part which is related to the word
        cat is the part used in the first part of the dm, without the final number aux
        aux is the number part
        """
        if lemma:
            x = "_"+lemma+"_"+cat
            if aux:
                return x+"_"+aux
            else:
                return x
        else:
            if aux:
                return cat + "_"+aux
            else:
                return cat

    @staticmethod
    def get_arg_signature(args):
        return "-".join([arg for (role, arg, attach) in args])


    def parse_semi_file(self,f):
        arg_reg=r"(?P<role>\S+)\s(?P<arg>\w+)(?P<attach>\s\{.*?\})?,?"
        with open(f,"r") as fin:
            for line in fin:
                if "predicates:" in line  or line == "\n":
                    continue
                else:
                    #_abaculus_n_1 : ARG0 x { IND + }.
                    semi_p= line.split(":")[0].lstrip().rstrip()
                    semi_a= line.split(":")[1].lstrip().rstrip()
                    lemma,cat,aux = SEMIReader.get_lem_cat_aux_from_semi_p(semi_p)
                    args = SEMIReader.get_arg_list_from_semi_a(semi_a)
                    arg_sign = SEMIReader.get_arg_signature(args)
                    dm_frame = SEMIReader.get_dm_frame(cat, args)
                    eds_label = SEMIReader.get_eds_label(lemma, cat, aux)
                    semi = SEMI(lemma, cat, aux, args, arg_sign, dm_frame, eds_label)
                    self.add_semi(semi)

    def add_semi(self, semi):
        """
        add cannonical lemma to possible set of words including for aliases of the words
        adding the Frame into lemma mapping
        """
        lemma = semi.lemma
        splits = lemma.split('+')
        if len(splits) > 1 and splits[1]!="":
            self.joints.add(" ".join(splits))
            compounds = splits+[MWE_END]
            past = ""
            for w in compounds:
                self.joints_map.setdefault(past[:-1],[]).append(w)
                past = past + w + "+"

        if lemma != "":
            # surface semi
            frames = self.surface_frames.setdefault(lemma,[])
            frames.append(semi)
        else:
            # abstract
            cat = semi.cat
            frames = self.abstract_frames.setdefault(semi.cat, [])
            frames.append(semi)

g_semi_reader = SEMIReader()

def main():
    logger = logging.getLogger("mrp")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    with open(semi_folder_path+"/semi_joint.txt", "w+") as fout:
        for i in g_semi_reader.joints:
            fout.write("{}\n".format(i))

    lemma_cnt = 0
    semi_cnt = 0
    for lemma, semis in g_semi_reader.surface_frames.items():
        lemma_cnt += 1
        semi_cnt += len(semis)
        logger.info("lemma = {}".format(lemma))
    logger.info("lemma_cnt={} semi_cnt={}, abstract_cnt={}".format(lemma_cnt, semi_cnt, len(g_semi_reader.abstract_frames)))

if __name__ == "__main__":
    main()
