#!/usr/bin/env python3.6
# coding=utf-8
'''

Some helper functions for storing and reading data

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-29

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-01
'''
import json,os,re
import pickle
from utility.top_utils.tree import *
from utility.mtool.graph import *
from utility.mtool.codec.mrp import read as mrp_read

class Pickle_Helper:

    def __init__(self, filePath):
        self.path = filePath
        self.objects = dict()

    def dump(self,obj,name):
        self.objects[name] = obj

    def save(self):
        f = open(self.path , "wb")
        pickle.dump(self.objects ,f,protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self):
        f = open(self.path , "rb")
        self.objects = pickle.load(f)
        f.close()
        return self.objects

    def get_path(self):
        return self.path


class Json_Helper:

    def __init__(self, filePath):
        self.path = filePath
        self.objects = dict()

    def dump(self,obj,name):
        self.objects[name] = obj

    def save(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        for name in self.objects:
            with open(self.path+"/"+name+".json", 'w+') as fp:
                json.dump(self.objects[name], fp)

    def load(self):
        files_path = folder_to_files_path(self.path,ends =".json")
        for f in files_path:
            name = f.split("/")
            with open(f) as data_file:
                data = json.load(data_file)
                self.objects[name] = data
        return self.objects

    def get_path(self):
        return self.path

# get all the files in the folder with a speficied file suffix, sort it to make it alphanumeric order
# os.lisdir is an order related to file added time
def folder_to_files_path(folder,ends =".txt"):
    files = os.listdir(folder )
    files.sort()
    files_path = []
    for f in files:
        if f.endswith(ends):
            files_path.append(folder+f)
          #  break
    return   files_path

def load_line(line,data):
    """
    load line for jamr comment format, to load all tokens, root, node, edge. and laign.
    """
    if "\t" in line:
        tokens = line[4:].split("\t")
    else:
        tokens = line[4:].split()
    if tokens[0] == "root": return

    if tokens[0] == "node":
        data["node"][tokens[1]] = tokens[2]
        if tokens.__len__() > 3:
            # only store the start of the alignment span
            data["align"][tokens[1]] = int(tokens[3].split("-")[0])
        return
    if tokens[0] == "edge":
        data["edge"][tokens[4],tokens[5]] = tokens[2]
        return
    data[tokens[0]] = tokens[1:]

def asserting_equal_length(data):
    assert len(data["tok"]) ==len(data["lem"]) , (  len(data["tok"]) ,len(data["lem"]),"\n",list(zip(data["tok"],data["lem"])) ,data["tok"],data["lem"])
    assert len(data["tok"]) ==len(data["ner"]) , (  len(data["tok"]) ,len(data["ner"]),"\n",list(zip(data["tok"],data["ner"])) ,data["tok"],data["ner"])
    assert len(data["tok"]) ==len(data["pos"]) , (  len(data["tok"]) ,len(data["pos"]),"\n",list(zip(data["tok"],data["pos"])) ,data["tok"],data["pos"])
    assert len(data["tok"]) ==len(data["anchors"]) , (  len(data["tok"]) ,len(data["anchors"]),"\n",list(zip(data["tok"],data["anchors"])) ,data["tok"],data["anchors"])

def load_text_jamr(filepath):
    """
    it is loading inputs and amr from the original amr file
    ### ::id xxxx
    ### ::snt xxxx
    ### ::tok xxx
    (a/amr-empty)
    """
    all_data = []
    with open(filepath,'r') as f:
        line = f.readline()
        while line != '' :
            while line != '' and not line.startswith("# ::") :
                line = f.readline()

            if line == "": return all_data

            data = {}
            data.setdefault("align",{})
            data.setdefault("node",{})
            data.setdefault("edge",{})
            # read tok, lem, pos, ner
            while line.startswith("# ::"):
                load_line(line.replace("\n","").strip(),data)
                line = f.readline()
            amr_t = ""
            while line.strip() != '' and not line.startswith("# AMR release"):
                amr_t = amr_t+line
                line = f.readline()
            data["amr_t"] = amr_t
            asserting_equal_length(data)
            all_data.append(data)
            line = f.readline()
    return all_data

def load_text_input(filepath):
    """
    it is loading inputs only from the original amr file
    ### ::id xxxx
    ### ::snt xxxx
    ### ::tok xxx
    (a/amr-empty)
    """
    all_data = []
    with open(filepath,'r') as f:
        line = f.readline()
        while line != '' :
            while line != '' and not line.startswith("# ::"):
                line = f.readline()

            if line == "": return all_data

            data = {}
            while line.startswith("# ::"):
                load_line(line.replace("\n","").strip(),data)
                line = f.readline()
            all_data.append(data)
            line = f.readline()
    return all_data


def genKey(graph):
    """
    mrp_amr, mrp_ucca, mrp_eds, mrp_psd, mrp_dm
    """
    return "mrp_"+graph.framework

def readFeaturesInputList(filepaths):
    """
    load all features from mrp input file into a list to keep the order
    """
    input_list = []
    for filepath in filepaths:
        with open(filepath, 'r') as fp:
            for graph, _ in mrp_read(fp):
                # here graph.framework is conllu by default
                data = {genKey(graph) : graph}
                # add unique example_id
                data["example_id"] = graph.id
                data["input_snt"] = graph.input
                # label is token
                data["tok"] = [node.label for node in graph.nodes]
                # 0 if lemma
                if "lemma" in graph.nodes[0].properties:
                    lemma_index = graph.nodes[0].properties.index("lemma")
                    data["lem"] = [node.values[lemma_index] for node in graph.nodes]

                if "xpos" in graph.nodes[0].properties:
                    xpos_index = graph.nodes[0].properties.index("xpos")
                    data["xpos"] = [node.values[xpos_index] for node in graph.nodes]

                if "upos" in graph.nodes[0].properties:
                    upos_index = graph.nodes[0].properties.index("upos")
                    data["upos"] = [node.values[upos_index] for node in graph.nodes]

                if "pos" in graph.nodes[0].properties:
                    pos_index = graph.nodes[0].properties.index("pos")
                    data["pos"] = [node.values[pos_index] for node in graph.nodes]

                if "ner" in graph.nodes[0].properties:
                    ner_index = graph.nodes[0].properties.index("ner")
                    data["ner"] = [node.values[ner_index] for node in graph.nodes]

                if "mwe" in graph.nodes[0].properties:
                    mwe_index = graph.nodes[0].properties.index("mwe")
                    data["mwe"] = [node.values[mwe_index] for node in graph.nodes]

                # if no anchors, it will be None.
                data["anchors"] = [node.anchors for node in graph.nodes]

                input_list.append(data)
    return input_list


def readFeaturesInput(filepaths):
    """
    load all features from mrp input file
    """
    input_dict = {}
    for filepath in filepaths:
        with open(filepath, 'r') as fp:
            for graph, _ in mrp_read(fp):
                # here graph.framework is conllu by default
                data = {genKey(graph) : graph}
                # add unique example_id
                data["example_id"] = graph.id
                data["input_snt"] = graph.input
                # label is token
                data["tok"] = [node.label for node in graph.nodes]
                # 0 if lemma
                if "lemma" in graph.nodes[0].properties:
                    lemma_index = graph.nodes[0].properties.index("lemma")
                    data["lem"] = [node.values[lemma_index] for node in graph.nodes]

                if "xpos" in graph.nodes[0].properties:
                    xpos_index = graph.nodes[0].properties.index("xpos")
                    data["xpos"] = [node.values[xpos_index] for node in graph.nodes]

                if "upos" in graph.nodes[0].properties:
                    upos_index = graph.nodes[0].properties.index("upos")
                    data["upos"] = [node.values[upos_index] for node in graph.nodes]

                if "pos" in graph.nodes[0].properties:
                    pos_index = graph.nodes[0].properties.index("pos")
                    data["pos"] = [node.values[pos_index] for node in graph.nodes]

                if "ner" in graph.nodes[0].properties:
                    ner_index = graph.nodes[0].properties.index("ner")
                    data["ner"] = [node.values[ner_index] for node in graph.nodes]

                if "mwe" in graph.nodes[0].properties:
                    mwe_index = graph.nodes[0].properties.index("mwe")
                    data["mwe"] = [node.values[mwe_index] for node in graph.nodes]

                # if no anchors, it will be None.
                data["anchors"] = [node.anchors for node in graph.nodes]

                input_dict[graph.id] = data
    return input_dict

def mergeWithAnnotatedGraphs(input_dict, filepaths):
    """
    read graph as the target
    """
    n = 0
    for filepath in filepaths:
        with open(filepath, 'r') as fp:
            for graph,_ in mrp_read(fp):
                # here graph.framework is conllu by default
                key = genKey(graph)
                if graph.id in input_dict:
                    input_dict[graph.id][key] = graph
                    n = n+1
    return n


def load_top_dataset(filepaths, dataset_name):
    tree_dataset = {}
    i = 0
    for filepath in filepaths :
        with open(filepath, "r") as fin, open(filepath+".bak", "w+") as fout:
            for tsv_line in fin:
                lines = tsv_line.split('\t')
                if len(lines) == 3:
                    snt = lines[0]
                    tokenized_snt = lines[1]
                    i = i +1
                    tree = Tree(lines[2])
                    tree_dataset[dataset_name+"_"+ str(i)] = {"input": tokenized_snt, "tree": tree}
                elif len(lines) == 5:
                    new_lines = []
                    snt = lines[2]
                    tokenized_snt = lines[2]
                    tree_str = lines[4].rstrip('\n')
                    tree = Tree(tree_str)
                    i = i +1
                    new_lines.append(lines[2])
                    new_lines.append(lines[2])
                    new_lines.append(tree_str)
                    tree_dataset[dataset_name+"_"+ str(i)] = {"input": tokenized_snt, "tree": tree}
                    fout.write("\t".join(new_lines))
                    fout.write("\n")
                else:
                    print("tsv_line:{}".format(lines))

    return tree_dataset
