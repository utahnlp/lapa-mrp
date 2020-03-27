#!/usr/bin/env python3.6
# coding=utf-8
'''

This reader reads all psd vallex file,
and add possible cannonical vallex lemma
to the corresponding copying dictionary of a word and aliases of the word

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-28
'''

import xml.etree.ElementTree as ET
from utility.psd_utils.PSDGraph import *
import re
from utility.constants import *
import logging

logger = logging.getLogger("mrp.psd")

def add_concept(lemmas_to_concept,le,con):

    if not le in lemmas_to_concept:
        lemmas_to_concept[le]= [con]
    else:
        lemmas_to_concept[le].append(con)

sense_reg=r"(f\d+.*)"

class VallexReader:
    def parse(self):
        """
        parse all the psd vallex frames
        """
        # for every word key, there is a set of fram, every frame is a sense
        self.frames = dict()
        self.word_ids = dict()
        self.non_sense_frames = dict()
        self.frame_all_args = dict()
        self.frame_oblig_args = dict()
        self.frame_lemmas = set()
        self.joints = set()
        self.joints_map = {}
        # for psd, only one file extised for vallex lexicon
        self.parse_file(self.frame_file_path)

    def __init__(self, file_path=vallex_file_path):
        self.frame_file_path = file_path
        self.parse()

    def parse_file(self,f):
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if child.tag == "body":
                # iterate every word
                for wordChild in child:
                    self.add_lemma(wordChild)

    @staticmethod
    def extract_sense_with_wordid(word_id, frame_id):
        """
        word id is the prefix of the frame_id
        """
        if word_id in frame_id:
            return frame_id.replace(word_id)
        else:
            logger.error("{} is not prefix of {}".format(word_id, frame_id))
            # when cannot be splitted, we just use the frame_id
            return frame_id

    def extract_word_and_sense(frame_id):
        """
        without using the lexicon, split by string match
        """
        splits = re.split(sense_reg, frame_id)
        word_id = splits[0]
        sense = splits[1]
        return word_id, sense

    def extract_sense_with_lemma(self,lemma, frame_id):
        """
        extract the lemma and sense, mot use the word_id, because it is not word
        # we only support the connected lemma, replace space with "_"
        """
        if lemma in self.word_ids:
            word_id = self.word_ids[lemma]
            sense = VallexReader.extract_sense_with_wordid(word_id, frame_id)
            return sense
        else:
            logger.error("{} is not in our vallex lexicon, use whole frame_id as sense ={}".format(lemma, frame_id))
            return frame_id


    def get_frame_id(self, lemma, sense):
        """
        given a lemma and sense, return the full frame id
        """
        if lemma in self.word_ids and sense in self.frames[lemma]:
            word_id = self.word_ids[lemma]
            frame_id = word_id + sense
        else:
            # lemma is not in the dictionary
            # try to find the most similar one
            logger.error("{} is not vallex dict".format(lemma))
            frame_id = "N/A"

        return frame_id

    def check_functor_in_oblig_args(self, frame_id, arg):
        if frame_id in self.frame_oblig_args:
            return arg in self.frame_oblig_args[frame_id]
        else:
            return False


    def add_lemma(self,node):
        """
        add cannonical amr lemma to possible set of words including for aliases of the words
        adding the Frame into lemma mapping
        """
        # heat_up is underscore for psd, 20088019
        lemma =  node.attrib["lemma"]
        word_id = node.attrib["id"]
        self.word_ids[lemma] = word_id
        self.frame_lemmas.add(lemma)
        self.frames.setdefault(lemma,[])
        # frame id is attaching some suffix frame id after word_id, {word_id}
        # we don't classify sense_id, just use a mapping here.
        # POS can be ignored, most of them are V,
        # 11 POS="A"
        # 5 POS="M"
        # 1 POS="N"
        # 4337 POS="V"
        splits = lemma.split("_")
        if len(splits) > 1:
            self.joints.add(" ".join(splits))
            compounds = splits+["<MWE_END>"]
            past = ""
            for w in compounds:
                self.joints_map.setdefault(past[:-1],[]).append(w)
                past = past + w + "_"

    #    self.frames[lemma] = set()
        for child in node:
            if child.tag == "valency_frames":
                for frame in child:
                    if frame.tag == "frame":
                        frame_id = frame.attrib["id"]
                        args = self.frame_oblig_args.setdefault(frame_id,[])
                        all_args = self.frame_all_args.setdefault(frame_id,[])
                        # we can use the whole thing as sense
                        x_word_id, sense = VallexReader.extract_word_and_sense(frame_id)
                        if x_word_id != word_id:
                            logger.error("{} != {}, extracted word_id from frameid is not equal to the original word_id".format(x_word_id, word_id))
                        add_concept(self.frames,lemma,sense)
                        for f_elements in frame:
                            if f_elements.tag == "frame_elements":
                                # get all of its fuctors
                                for elem in f_elements:
                                    if elem.tag == "element":
                                        functor = elem.attrib["functor"]
                                        all_args.append(functor)
                                        if "type" in elem.attrib and elem.attrib["type"] == "oblig":
                                            args.append(functor)
                                    elif elem.tag == "element_alternation":
                                        # see w1255f4
                                        for s_elem in elem:
                                            if s_elem.tag == "element":
                                                functor = s_elem.attrib["functor"]
                                                all_args.append(functor)
                                                if "type" in s_elem.attrib and s_elem.attrib["type"] == "oblig":
                                                    args.append(functor)


    def get_frames(self):
        return self.frames

g_vallex_reader = VallexReader()

def main():
    with open(semi_folder_path+"/vallex_joint.txt", "w+") as fout:
        for i in g_vallex_reader.joints:
            fout.write("{}\n".format(i))

    logger.info("len(self.frame_lemma)={}".format(len(f_r.frame_lemmas)))


if __name__ == "__main__":
    main()
