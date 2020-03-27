#!/usr/bin/env python3.6
# coding=utf-8
'''

Combine multiple mrp data files in the same directory into a single one
Need to specify folder containing all subfolders of training, dev and test

Then extract features for futher process based on stanford core nlp tools

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-06-01

# directly read from udpipe mrp files, and then adding extra annotations, such as ner
@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-01
'''
import argparse

from pycorenlp import StanfordCoreNLP
from utility.constants import *
from utility.data_helper import *
from utility.mtool.codec.mrp import read as mrp_read
from utility.mtool.codec.amr import convert_amr_id
from parser.AMRProcessors import *
from parser.DMProcessors import *
from parser.PSDProcessors import *
from parser.EDSProcessors import *
from parser.UCCAProcessors import *
import logging

logger = logging.getLogger("mrp.preprocessing")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def featureExtract(src_text,whiteSpace=False):
    """
    Using stanford nlp url to extract features from text
    whiteSpace means only split workds when there is a whitespace, it can be used to keep existed tokenization
    """
    data = {}
    output = nlp.annotate(src_text.strip(), properties={
        'annotators': "tokenize,ssplit,ner",
        #"tokenize.options":"splitHyphenated=false,normalizeParentheses=false,untokenizable='allKeep'",
      	#"tokenize.whitespace": whiteSpace,
        "tokenize.language": "Whitespace" if whiteSpace else "English",
        'ssplit.isOneSentence': True,
        'outputFormat': 'json'
    })
    snt = output['sentences'][0]["tokens"]
    data["input_snt"] = src_text
    data["ner"] = []
    data["tok"] = []
    data['pos'] = []
    data['lem'] = []
    data['anchors'] = []
    for snt_tok in snt:
        data["ner"].append(snt_tok['ner'])
        data["tok"].append(snt_tok['word'])
        # first add anchors as a dictionary here.
        data["anchors"].append([{'from': snt_tok['characterOffsetBegin'], 'to': snt_tok['characterOffsetEnd']}])
        data["pos"].append(snt_tok['pos'])
        data["lem"].append(snt_tok['lemma'])

    data['mwe'] = ['O'] * len(data["tok"])
    assert len(data["ner"]) ==len(data["tok"]) , (  len(data["tok"]) ,len(data["ner"]),"\n",list(zip(data["tok"],data["ner"])) ,data["tok"],data["ner"])
    #  if whiteSpace is False:
    #      return self.featureExtract(" ".join(data["tok"]),True)
    return data

def write_features(filepath):
    if ".mrp" in opt.companion_suffix:
        write_features_mrp(filepath)
    elif ".amr" in opt.companion_suffix:
        write_features_amr(filepath)
    else:
        raise NotImplementedError("Not support for reading {}".format(opt.companion_suffix))

def write_features_amr(filepath):
    out = filepath.split(opt.companion_suffix)[0] + ".mrp_conllu_pre_processed"
    logger.info("processing "+filepath)
    with open(out,'w') as out_f:
        with open(filepath,'r') as f:
            n = 0
            line = f.readline()
            example_id = ""
            while line != '' :
                if line.startswith("# ::id"):
                    # ::id bc.cctv_0000.1 ::date 2012-12-11T19:08:03# ::id bc.cctv_0000.1 ::date 2012-12-11T19:08:03
                    example_id = line[7:].split(' ')[0]
                    try:
                        example_id = convert_amr_id(example_id)
                    except:
                        pass
                elif line.startswith("# ::snt") or line.startswith("# ::tok"):
                    n = n+1
                    if n % 500 ==0:
                        logger.info(n)
                    text = line[7:].rstrip("\n")
                    if input_preprocessor:
                        # when using combining, only adding combined nodes, withouting using udpipe and edges.
                        if opt.token_combine:
                            data = input_preprocessor.preprocess(text, whiteSpace=False) #phrase from fixed joints.txt file
                        else:
                            data = input_preprocessor.featureExtract(text, whiteSpace=False) #phrase from fixed joints.txt file
                    else:
                        data = featureExtract(text, whiteSpace=True)
                    # constructing a new graph
                    assert example_id != "", "empty example_id in line={}".format(line)
                    new_graph = Graph(example_id, 2 , "amr")
                    for i in range(len(data['tok'])):
                        if "mwe" in data:
                            new_graph.add_node(i, label=data['tok'][i], properties=["lemma","pos","ner","mwe"], values=[data['lem'][i], data['pos'][i], data['ner'][i],data['mwe'][i]])
                        else:
                            new_graph.add_node(i, label=data['tok'][i], properties=["lemma","pos","ner"], values=[data['lem'][i], data['pos'][i], data['ner'][i]])
                    new_graph.add_input(text)
                    out_f.write(json.dumps(new_graph.encode(), indent=None, ensure_ascii = False))
                    out_f.write("\n")
                    example_id = ""
                elif not line.startswith("# AMR release; "):
                    pass
                line = f.readline()
    logger.info("done processing "+filepath)
    logger.info(out +" is generated")

def write_features_mrp(filepath):
    """
    write preprocessed features like tok, lem, pos, ner in mrp_conllupre_prossed
    """
    out = filepath.split(opt.companion_suffix)[0] + ".mrp_conllu_pre_processed"
    logger.info("processing "+filepath)
    with open(out,'w') as out_f:
        with open(filepath, 'r') as in_file:
            n = 0
            for graph,_  in mrp_read(in_file):
                n = n + 1
                if n % 500 == 0:
                    logger.info(n)
                    # only add a ner feature from that
                #if graph.id not in ['bolt-eng-DF-170-181118-8875443_0097.13','bolt-eng-DF-170-181103-8882248_0335.5']:
                #    continue
                tokenized_text = ' '.join([node.label for node in graph.nodes])
                text = graph.input
                text = text.replace(u"\u0085",u"\00A0").replace("%20",u"\00A0")
                tokenized_text = tokenized_text.replace(u"\u0085",u"\00A0").replace("%20",u"\00A0")
                if opt.frame == 'amr' or opt.frame == 'ucca':
                    data = input_preprocessor.preprocess(text, whiteSpace=False, token_combine = opt.token_combine) #phrase from fixed joints.txt file
                    # constructing a new graph
                    new_graph = Graph(graph.id, graph.flavor,graph.framework)
                    new_graph.add_input(text)
                    for i in range(len(data['tok'])):
                        if "mwe" in data:
                            new_graph.add_node(i, label=data['tok'][i], properties=["lemma","pos","ner","mwe"], values=[data['lem'][i], data['pos'][i], data['ner'][i],data['mwe'][i]], anchors=data['anchors'][i])
                        else:
                            new_graph.add_node(i, label=data['tok'][i], properties=["lemma","pos","ner"], values=[data['lem'][i], data['pos'][i], data['ner'][i]], anchors=data['anchors'][i])
                    out_f.write(json.dumps(new_graph.encode(), indent=None, ensure_ascii = False))
                    out_f.write("\n")
                else:
                    # use white space and only use ner for extra
                    data = input_preprocessor.preprocess(tokenized_text, whiteSpace=True, token_combine = opt.token_combine) 
                    assert len(data['ner']) == len(graph.nodes), "preprocess data length is not equal to the input in {}, {}".format(graph.encode(), data)
                    assert len(data['mwe']) == len(graph.nodes), "preprocess data length is not equal to the input in {}, {}".format(graph.encode(), data)
                    for node in graph.nodes:
                        i = node.properties.index('xpos');
                        node.set_property('pos', node.values[i])
                        node.set_property('ner', data['ner'][node.id])
                        node.set_property('mwe', data['mwe'][node.id])
                    # write back ner
                    out_f.write(json.dumps(graph.encode(), indent=None, ensure_ascii = False))
                    out_f.write("\n")

    logger.info("done processing "+filepath)
    logger.info(out +" is generated")

def combine_arg():
    # To use this preprocessing, the input is in mrp format
    parser = argparse.ArgumentParser(description='preprocessing.py, input is mrp format, which is read from conllu format, and written into mrp format.')

    ## Data options
    # combine all .txt files
    parser.add_argument('--suffix', default=".mrp_conllu", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--companion_suffix', default=".mrp_conllu", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--folder', default="", type=str ,
                        help="""the folder""")
    parser.add_argument('--build_folder', default="", type=str ,
                        help="""the folder for build preprocessed data""")
    parser.add_argument('--frame', default="", type=str,
                        help="""weather to do amr preprocess""")
    parser.add_argument('--token_combine', default=False, type=bool,
                        help="""weather to combine tokens, now it is mainly amr""")
    return parser


parser = combine_arg()

opt = parser.parse_args()

nlp = StanfordCoreNLP(core_nlp_url)

if opt.frame == "amr":
    input_preprocessor = AMRInputPreprocessor(opt, core_nlp_url)
elif opt.frame == "dm":
    input_preprocessor = DMInputPreprocessor(opt, core_nlp_url)
elif opt.frame == "psd":
    input_preprocessor = PSDInputPreprocessor(opt, core_nlp_url)
elif opt.frame == "eds":
    input_preprocessor = EDSInputPreprocessor(opt, core_nlp_url)
elif opt.frame == "ucca":
    input_preprocessor = UCCAInputPreprocessor(opt, core_nlp_url)
else:
    input_preprocessor = AMRInputPreprocessor(opt, core_nlp_url)

trainFolderPath = opt.build_folder + "/training/"
trainingCompanionFilesPath = folder_to_files_path(trainFolderPath, opt.companion_suffix)

devFolderPath = opt.build_folder + "/dev/"
devCompanionFilesPath = folder_to_files_path(devFolderPath, opt.companion_suffix)

testFolderPath = opt.build_folder + "/test/"
testCompanionFilesPath = folder_to_files_path(testFolderPath, opt.companion_suffix)

for f in trainingCompanionFilesPath:
    write_features(f)

for f in devCompanionFilesPath:
    write_features(f)

for f in testCompanionFilesPath:
    write_features(f)
