#!/usr/bin/env python3.6
# coding=utf-8
'''

AMRParser for producing amr graph from raw text
AMRDecoder for decoding deep learning model output into actual AMR concepts and graph
AMRInputPreprocessor for extract features based on stanford corenlp
@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30

# This AMRProcessors are used as a luanch for preprocessing the input raw sentence and parse it
# Ignore this file, if the preprocessed data has been given
# preprocessing for AMR is import for later alignments
# But it may be not good for multi-task learning.
@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-01
'''
from utility.amr_utils.AMRStringCopyRules import  *
from utility.amr_utils.AMRReCategorization import  *
from parser.Dict import seq_to_id
from utility.constants import core_nlp_url
from utility.amr_utils.amr import *
import networkx as nx
from utility.mtool.graph import *
import random
import string

from src import *
from parser.modules.helper_module import myunpack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence
from parser.DataIterator import DataIterator,rel_to_batch
from pycorenlp import StanfordCoreNLP
import logging

logger = logging.getLogger("amr.AMRProcessors")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class AMRInputPreprocessor(object):
    """
    A feature extractor for amr input
    """
    def __init__(self, opt, url = core_nlp_url):
        self.nlp = StanfordCoreNLP(url)
        self.opt = opt
        self.joints_map = self.readJoints()
        self.number_texts = {"hundred", "thousand", "million", "billion", "trillion", "hundreds", "thousands",
			"millions", "billions", "trillions"}
        self.slashedNumber =  re.compile(r'-*\d+-\d+')

    def readJoints(self):
        joints_map = {}
        with open(self.opt.build_folder+"dicts/joints.txt",'r') as f:
            line = f.readline()
            while line.strip() != '':
                line = f.readline()
                compounds = line.split()
                past = ""
                for w in compounds:
                    joints_map.setdefault(past[:-1],[]).append(w)
                    past = past + w + "-"
        return joints_map

    def combine_number(self,data):
    #combine phrase e.g. :  make up
        def combinable_number(n1,n2):
            return n2 in self.number_texts and n1 != "-"
        def combinable(i,m):
            return len(lemma) > 0 and m == "CD"\
                    and pos[-1] =="CD" and combinable_number(lemma[-1], data["lem"][i])
        lemma = []
        ner = []
        tok = []
        pos = []
        anchors = []

        for i, m in enumerate(data["pos"]):
            if combinable(i,m) :
                lemma[-1] = lemma[-1] +"," + data["lem"][i]
                tok[-1] = tok[-1] + "," + data["tok"][i]
                # data["anchors"].append([{'from': snt_tok['beginChar'], 'to': snt_tok['endChar']}])
                anchors[-1].extend(data["anchors"][i])
                pos[-1] = "CD"
        #        ner[-1] = ner[-1]
            else:
                lemma.append(data["lem"][i])
                tok.append(data["tok"][i])
                # extends anchors means two dict in an array
                # every dictionary is corresponing to an original token
                anchors.append(data["anchors"][i])
                pos.append(data["pos"][i])
                ner.append(data["ner"][i])

        data["lem"] = lemma
        data["ner"] = ner
        data["pos"] = pos
        data["tok"] = tok
        data["anchors"] = anchors
        return data

    def tag_url_and_split_number(self,data):
        lemma = []
        ner = []
        tok = []
        pos = []
        anchors = []

        for i, le in enumerate(data["lem"]):
            if "http" in le or "www." in le:
                ner.append("URL")
                lemma.append(data["lem"][i])
                tok.append(data["tok"][i])
                anchors.append(data["anchors"][i])
                pos.append(data["pos"][i])

            elif re.match(self.slashedNumber, le) and data["ner"][i] == "DATE":
                les = le.replace("-"," - ").split()
                toks = data["tok"][i].replace("-"," - ").split()
                # here original anchors will also be splitted
                splitted_anchors = []
                start = data["anchors"][i][0]["from"]
                for x in toks:
                    end = start + len(x)
                    splitted_anchors.append([{'from': start,'to': end}])
                    start = end
                assert len(les) == len(toks),data
                for  l in les:
                    if l != "-":
                        pos.append(data["pos"][i])
                        ner.append(data["ner"][i])
                    else:
                        pos.append(":" )
                        ner.append("0")
                lemma = lemma + les
                tok = tok + toks
                # anchors to original sentence will not be changed.
                anchors = anchors + splitted_anchors
            else:
                ner.append(data["ner"][i])
                lemma.append(data["lem"][i])
                tok.append(data["tok"][i])
                pos.append(data["pos"][i])
                anchors.append(data['anchors'][i])

        data["lem"] = lemma
        data["ner"] = ner
        data["pos"] = pos
        data["tok"] = tok
        data["anchors"] = anchors
        return data

    def combine_phrase(self,data):
        # for combined phrase, the token is not good for mrp, because every token maybe annotated for other 
        #combine phrase e.g. :  make up
        lemma = []
        ner = []
        tok = []
        pos = []
        anchors = []
        skip = False
        for i ,le in enumerate(data["lem"]):
            if skip:
                skip = False
            elif len(lemma) > 0 and le in self.joints_map.get( lemma[-1] ,[]) :
                    lemma[-1] = lemma[-1] +"-"+le
                    tok[-1] = tok[-1]+ "-" + data["tok"][i]
                    # data["anchors"].append([{'from': snt_tok['beginChar'], 'to': snt_tok['endChar']}])
                    anchors[-1].extend(data["anchors"][i])
                    pos[-1] = "COMP"
                    ner[-1] = "0"
            elif len(lemma) > 0 and le == "-" and i < len(data["lem"])-1 \
                and data["lem"][i+1] in self.joints_map.get( lemma[-1] ,[]):
                lemma[-1] = lemma[-1] +"-"+data["lem"][i+1]
                tok[-1] = tok[-1]+ "-" + data["tok"][i+1]
                anchors[-1].extend(data["anchors"][i+1])
                pos[-1] = "COMP"
                ner[-1] = "0"
                skip = True
            else:
                lemma.append(le)
                tok.append(data["tok"][i])
                pos.append(data["pos"][i])
                ner.append(data["ner"][i])
                anchors.append(data["anchors"][i])

        data["lem"] = lemma
        data["ner"] = ner
        data["pos"] = pos
        data["tok"] = tok
        data["anchors"] = anchors
        return data


    def featureExtract(self,src_text,whiteSpace=False):
        """
        Using stanford nlp url to extract features from AMR text
        whiteSpace means only split workds when there is a whitespace, it can be used to keep existed tokenization
        """
        data = {}
        output = self.nlp.annotate(src_text.strip(), properties={
        'annotators': "tokenize,ssplit,pos,lemma,ner",
        "tokenize.options":"splitHyphenated=true,normalizeParentheses=false",
		"tokenize.whitespace": whiteSpace,
        'ssplit.isOneSentence': True,
        'outputFormat': 'json'
    })
        snt = output['sentences'][0]["tokens"]
        data["ner"] = []
        data["tok"] = []
        data["lem"] = []
        data["pos"] = []
        data["anchors"] = []
        for snt_tok in snt:
            data["ner"].append(snt_tok['ner'])
            data["tok"].append(snt_tok['word'])
            # first add anchors as a dictionary here.
            data["anchors"].append([{'from': snt_tok['characterOffsetBegin'], 'to': snt_tok['characterOffsetEnd']}])
            data["lem"].append(snt_tok['lemma'])
            data["pos"].append(snt_tok['pos'])
     #   if whiteSpace is False:
     #       return self.featureExtract(" ".join(data["tok"]),True)
        asserting_equal_length(data)
        return data

    def preprocess(self,src_text, whiteSpace=False, token_combine = True):
        data = self.featureExtract(src_text, whiteSpace)
        if token_combine:
            data = self.combine_phrase(data) #phrase from fixed joints.txt file
            data = self.combine_number(data)
            data = self.tag_url_and_split_number(data)
            asserting_equal_length(data)
        return data

class AMRParser(object):
    """
    AMR Parser class
    """
    def __init__(self, opt,dicts):
        self.decoder = AMRDecoder(opt,dicts)
        self.model, _, _ = load_old_model(dicts,opt,True)[0]
        self.opt = opt
        self.feature_extractor = AMRInputPreprocessor(opt,core_nlp_url)
        self.dicts = dicts
        self.decoder.eval()
        self.model.eval()

    def feature_to_torch(self,all_data):
        for i, data in enumerate(all_data):
            if "example_id" not in data:
                data["example_id"] = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(8)])
            data["snt_id"] = seq_to_id(self.dicts ["word_dict"],data["tok"])[0]
            data["lemma_id"] = seq_to_id(self.dicts ["lemma_dict"],data["lem"])[0]
            data["pos_id"] =  seq_to_id(self.dicts ["pos_dict"],data["pos"])[0]
            data["ner_id"] = seq_to_id(self.dicts ["ner_dict"],data["ner"])[0]
        # when using all_data, batch_size is all
        if self.opt.bert_model:
            data_iterator = BertDataIterator([],self.opt,self.dicts["amr_rel_dict"],all_data=all_data)
            order,idsBatch,srcBatch,src_charBatch, sourceBatch,srcBertBatch,srcBertIndexBatch = data_iterator[0]
        else:
            data_iterator = DataIterator([],self.opt,self.dicts["amr_rel_dict"],all_data=all_data)
            order,idsBatch,srcBatch,src_charBatch, sourceBatch = data_iterator[0]
            srcBertBatch = None
            srcBertIndexBatch = None
        return order,idsBatch,srcBatch, src_charBatch, sourceBatch,srcBertBatch,srcBertIndexBatch,data_iterator

    def parse_batch_preprocessed_data(self, preprocessed_input_examples, set_wiki, normalize_mod):
        order,idsBatch,srcBatch, src_charBatch, sourceBatch,srcBertBatch,srcBertIndexBatch,data_iterator = self.feature_to_torch(preprocessed_input_examples)

        probBatch, src_enc = self.model(srcBatch, src_charBatch, rel=False, bertBatch=srcBertBatch, bertIndexBatch = srcBertIndexBatch)

        amr_pred_seq,concept_batches,aligns_raw,dependent_mark_batch = self.decoder.probAndSourceToConcepts(sourceBatch,srcBatch, src_charBatch, probBatch,getsense = True )

        amr_pred_seq = [ [(uni.cat,uni.le,uni.aux,uni.sense,uni)  for uni in seq ] for  seq in amr_pred_seq ]


        rel_batch,aligns = rel_to_batch(concept_batches,aligns_raw,data_iterator,self.dicts)
        rel_prob,roots = self.model((rel_batch,srcBatch, src_charBatch, src_enc,aligns), rel=True, bertBatch=srcBertBatch)
        graphs,rel_triples  =  self.decoder.relProbAndConToGraph(concept_batches, sourceBatch, rel_prob,roots,(dependent_mark_batch,aligns_raw),True,set_wiki,normalizeMod=normalize_mod)
        batch_out = [0]*len(graphs)

        batch_mrp_graphs = [0]*len(graphs)
        for i,data in enumerate(zip(idsBatch, sourceBatch,amr_pred_seq,concept_batches,rel_triples,graphs)):
            example_id, source,amr_pred,concept, rel_triple,graph= data
            mrp_graph, predicated_graph = graph_to_mrpGraph(example_id, graph, normalizeMod = True, flavor=2, framework="amr", sentence=" ".join(source[0]))

            out = []
            out.append( "# ::id "+ example_id +"\n")
            out.append( "# ::tok "+" ".join(source[0])+"\n")
            out.append(  "# ::lemma "+" ".join(source[1])+"\n")
            out.append(  "# ::pos "+" ".join(source[2])+"\n")
            out.append(  "# ::ner "+" ".join(source[3])+"\n")
            out.append(  self.decoder.nodes_jamr(graph))
            out.append(  self.decoder.edges_jamr(graph))
            out.append( predicated_graph)
            batch_out[order[i]] = "".join(out)+"\n"
            batch_mrp_graphs[order[i]] = mrp_graph
        return batch_mrp_graphs, batch_out

    def parse_one_preprocessed_data(self, preprocessed_input_example, set_wiki, normalize_mod):
        return self.parse_batch_preprocessed_data([preprocessed_input_example], set_wiki, normalize_mod)

    def parse_batch(self,src_text_batch, set_wiki, normalize_mod):
        all_data =[ self.feature_extractor.preprocess(src_text) for src_text in src_text_batch ]
        return self.parse_batch_preprocessed_data(all_data, set_wiki, normalize_mod)

    def parse_one(self,src_text, set_wiki, normalize_mod):
        return self.parse_batch([src_text], set_wiki, normalize_mod)

def mark_edge(n1,n2,edge,edges):
    edges.add((n1,n2,edge))
    inverse = get_inversed_edge(edge)
    edges.add((n2,n1,inverse))

# ::node	0.1.1.1.1	kilometer	26-27
def sub_to_amr(G,visited,n,visited_edges,d=1):
    uni = G.node[n]["value"]
    if uni.is_constant():
        if uni.cat == Rule_String:
            return "\""+ uni.le+"\"",visited,visited_edges
        return uni.le,visited,visited_edges
    s = "("+str(n)+" /"+" "+ uni.le+uni.sense

    to_be_visted_edge = set()  #to_be_visted_edges around this node
    to_be_visted = set() #to be visited nodes around this node
    for nb,edge_datas in G[n].items():
        # problem here.
        for key, edge_data in edge_datas.items():
            if not nb in visited :
                to_be_visted.add(nb)
            if (n,nb,edge_data["role"]) not in visited_edges:
                mark_edge(n,nb,edge_data["role"],to_be_visted_edge)
    for nb,edge_datas in G[n].items():
        for key, edge_data in edge_datas.items():
            #if ((n,nb,edge_data["role"]) in to_be_visted_edge) and ((n, nb, edge_data["role"]) not in visited_edges):
            if (n,nb,edge_data["role"]) in to_be_visted_edge:
                forced = "force_connect" in edge_data
                append =  "" # "!" if  forced else ""
                mark_edge(n,nb,edge_data["role"],visited_edges)
                s += "\n"+"    "*d+edge_data["role"]+append+" "

                if nb in visited and nb not in to_be_visted:
                    uni = G.node[nb]["value"]
                    if uni.is_constant():
                        if uni.cat == Rule_String:
                            s +=  "\""+ uni.le+"\""
                        else:
                            s +=  uni.le
                    else:
                        s += str(nb)
                else:
                    ss,visited,visited_edges = sub_to_amr(G,visited.union(to_be_visted),nb, visited_edges.union(to_be_visted_edge),d+1)
                    s += ss
    s += ")"
    return s,visited,visited_edges

def my_contracted_nodes(G,n1,n2,self_loops=False):
    d = G.node[n1]
    # Pay attention, when contact,the key information will be lost. try to fixed that after contacting
 #   print ("before",G.node[n1])
    G = nx.contracted_nodes(G, n1,n2,self_loops=self_loops)
 #   print ("after",G.node[n1])
    for k in d:
        G.node[n1][k] = d[k]
 #   print ("finally",G.node[n1])

    # after n2 get contracted, change all the keys for un0
    H = G.copy()
    for un0, un1, ori_key, edge_data in G.edges(keys=True, data=True):
        if (un0 != n1 and un1 != n1):
            continue
        else:
            required_key = edge_data["role"]
            H.remove_edge(un0, un1, key=ori_key)
            H.add_edge(un0, un1, key=required_key, role=required_key)
            for akey, avalue in edge_data.items():
                H.edges[un0, un1, required_key][akey] = avalue

    #logger.info("H.edges: {}".format(H.edges(keys=True,data=True)))
    return H

#collapsing person nodes when its' likely its' the same person being activated by NER and other role.
#e.g. President Donald Trump might evoke person node twice.
def contract_graph(G):
    NAMES = {}
    for n in G.node:
        if not G.node[n]["value"].cat == Rule_Frame:
            continue
        person = []
        for nb, edges_data in G[n].items():
            if G.node[nb]["value"].le == "person":
                for key, edge in edges_data.items():
                    if edge["role"] == ":ARG0":
                        person.append(nb)
                        break

        if len(person) < 2: continue
        p1 = person[0]
        for pi in  person[1:]:
            G = my_contracted_nodes(G,p1,pi)
        return contract_graph(G)


    for n in G.node:
        if not G.node[n]["value"].le == "name":
            continue
        top = None
        topn = None
        names = []
        namesn = []
        for nb,edges_data in G[n].items():
            for key, edge in edges_data.items():
                if edge["role"] == ":name-of":
                    top = G.node[nb]["value"]
                    topn = nb
                elif edge["role"].startswith(":op"):
                    names.append((edge["role"][2:],G.node[nb]["value"]))
                    namesn.append((edge["role"][2:],nb))
        ner =  (top,tuple(sorted(names,key=lambda x: x[0])))
        namesn = sorted(namesn,key=lambda x: x[0])
        if ner in NAMES:
            G = my_contracted_nodes(G, NAMES[ner][0],n)
            if top is None: continue
            G = my_contracted_nodes(G, NAMES[ner][1],topn)
            for s1,s2 in zip(namesn,NAMES[ner][2]):
                if s1[1] in G and s2[1] in G:
                    G = my_contracted_nodes(G,s2[1],s1[1])
            return contract_graph(G)
        else:
            NAMES[ner] = (n,topn,namesn)

    for n in G.node:
        for n1,edge1_data in G[n].items():
            for n2,edge2_data in G[n].items():
                if str(n1) < str(n2) and G.node[n1]["value"] ==  G.node[n2]["value"] and G.node[n1]["value"].is_constant():
                    for key1, edge1 in edge1_data.items():
                        for key2, edge2 in edge2_data.items():
                            if edge1["role"] == edge2["role"]:
                                return contract_graph(my_contracted_nodes(G, n1,n2))
    return G
#Turning data structure to AMR text for evaluation
def graph_to_amr(G, sentence = None):
    top = BOS_WORD
    root = list(G[top].keys())[0]
    uni = G.node[root]["value"]
    if uni.is_constant():
        G.node[root]["value"].cat = Rule_Concept
        G.node[root]["value"].le = G.node[root]["value"].le.strip(":").strip("/")
        if  G.node[root]["value"].le == "":
            G.node[root]["value"].le = "amr-unintelligible"
    s,visited,visited_edges =  sub_to_amr(G,{top,root},root,{(top,root,":top"),(root,top,":top-of")})
    for n in G.node:
        if not  n in visited:
            logger.info("node missing : %s", n)
            logger.info("nearbour : %s", G.node[n])
            for n in G.node:
                logger.info("nodes: %s, %s, %s", n,G.node[n],G.node[n]["value"])
            logger.info("visited nodes %s",visited)
            logger.info("edges %s",G.edges)
            logger.info("visited edges %s",visited_edges)
            logger.info("amr: %s, sentence: %s", s,sentence)
            break
      #      assert False
    for e in G.edges:
        if not  (e[0],e[1],G.edges[e]["role"]) in visited_edges:
            logger.info("edge missing %s %s %s",e[0],e[1],G.edges[e]["role"])
            logger.info("all nodes %s",G.nodes)
            logger.info("visited nodes %s",visited)
            logger.info("edges %s",G.edges)
            logger.info("visited edges %s",visited_edges)
            logger.info("amr: %s, sentence: %s", s,sentence)
            break
    return s

#decoder requires copying dictionary, and recategorization system
#full_tricks evoke some deterministic post processing, which might be expensive to compute
class AMRDecoder(object):
    def __init__(self, opt,dicts, frame = "amr"):
        """
        init the model for AMR Decoder in pytorch
        Especially, it relys on the dicts/vocabularies for tokens and labels. 
        """
        self.frame = frame
        self.opt = opt
        self.dicts = dicts
        # high frequency lemmas
        self.n_high = len(dicts["amr_high_dict"])
        # string rules for matching and alignements
        self.rl = AMRRules()
        with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
        # load the rules, which generated by rule_system_build
        self.rl.load(opt.build_folder+"dicts/amr_rule_f"+with_jamr)
        # init the recategorizer
        self.fragment_to_node_converter = AMRReCategorizor(from_file=True,path=opt.build_folder+"dicts/graph_to_node_dict_extended"+with_jamr,training=False,ner_cat_dict=dicts["amr_aux_dict"])
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @staticmethod
    def graph_to_mrpGraph(id, G, normalizeMod = True, flavor = None, framework = None, sentence = None):
        special_nodes = []
        attributes = []
        invalid_edges = []
        all_forward_edges = []
        mrp_graph = Graph(id, flavor = flavor, framework=framework)
        top = BOS_WORD
        root = list(G[top].keys())[0]
        uni = G.node[root]["value"]
        # when root is constant, not use attributes
        if uni.is_constant():
            G.node[root]["value"].cat = Rule_Concept
            G.node[root]["value"].le = G.node[root]["value"].le.strip(":").strip("/")
            if  G.node[root]["value"].le == "":
                G.node[root]["value"].le = "amr-unintelligible"
        s,visited,visited_edges =  sub_to_amr(G,{top,root},root,{(top,root,":top"),(root,top,":top-of")})

        # nodes
        for n in G.node:
            if n in visited:
                # n is involved in amr graph
                # add it into graph.nodes
                uni = G.node[n]["value"]
                if not uni.is_constant() and n != top:
                    # only add node that is not constant and not top
                    node_label = uni.le + uni.sense
                    node = mrp_graph.add_node(label = node_label, properties = None, values = None, anchors=None, top=False)
                    G.node[n]["mrp_node"] = node
                else:
                    special_nodes.append(uni)
                    continue
            else:
                logger.info("node missing : %s", n)
                logger.info("nearbour : %s", G.node[n])
                for n in G.node:
                    logger.info("nodes: %s, %s, %s", n,G.node[n],G.node[n]["value"])
                logger.info("visited nodes %s",visited)
                logger.info("edges %s",G.edges)
                logger.info("visited edges %s",visited_edges)
                logger.info("amr: %s, sentence: %s", s,sentence)

        assert (len(special_nodes) + len(mrp_graph.nodes) == len(visited)), "visited({}) != special({}) + concept_nodes({})".format(len(visited), len(special_nodes), len(mrp_graph.nodes))
        # edges
        for e in G.edges:
            if not (e[0],e[1],G.edges[e]["role"]) in visited_edges:
                logger.info("edge missing %s %s %s",e[0],e[1],G.edges[e]["role"])
                logger.info("all nodes %s",G.nodes)
                logger.info("visited nodes %s",visited)
                logger.info("edges %s",G.edges)
                logger.info("visited edges %s",visited_edges)
                logger.info("amr: %s, sentence: %s", s,sentence)
            else:
                # if visited:, both direction will be visited,  we only add normalized edges
                role = G.edges[e]["role"]
                lab = role[1:]
                uni0 = G.node[e[0]]["value"]
                uni1 = G.node[e[1]]["value"]
                if is_inversed_edge(role):
                    continue
                elif role == ":top":
                    # then uni1 should be root node, no relation or attributes toadd
                    assert "mrp_node" in G.node[e[1]], "mrp_node {} not created, but visited and not constant".format(uni1)
                    G.node[e[1]]["mrp_node"].is_top = True
                    attributes.append((uni0, role, uni1))
                    all_forward_edges.append((e,uni0, role, uni1))
                else:
                    all_forward_edges.append((e,uni0, role, uni1))
                    # only count forward relation, uni0 must be not constant
                    if uni0.is_constant() or uni1.is_constant():
                        if uni0.is_constant() and uni1.is_constant():
                            invalid_edges.append((uni0, role, uni1))
                            logger.error("no role should exist between two constant nodes, skipped")
                        elif not uni0.is_constant():
                            # uni0 is not constant, uni1 is constant
                            assert "mrp_node" in G.node[e[0]], "mrp_node {} not created, but visited and not constant".format(uni0)
                            if uni1.cat == Rule_String:
                                # attribute_value = "\""+uni1.le+"\""
                                attribute_value = uni1.le
                            else:
                                attribute_value = uni1.le

                            G.node[e[0]]["mrp_node"].set_property(lab, attribute_value)
                            attributes.append((uni0, role, uni1))
                        else:
                            # uni1 is not constant, uni0 is constant
                            invalid_edges.append((uni0, role, uni1))
                            logger.error("constant uni0 {}  should not has forward rel{}, skipped".format(uni0, role))
                    else:
                        assert "mrp_node" in G.node[e[0]], "mrp_node {} not created, but visited and not constant".format(uni0)
                        assert "mrp_node" in G.node[e[1]], "mrp_node {} not created, but visited and not constant".format(uni1)
                        src = G.node[e[0]]["mrp_node"].id
                        tgt  = G.node[e[1]]["mrp_node"].id
                        # we only add normalized relation
                        if normalizeMod and lab == "domain":
                            mrp_graph.add_edge(tgt, src, "mod")
                        else:
                            mrp_graph.add_edge(src, tgt, lab)

        if len(attributes) != len(special_nodes)-len(invalid_edges):
            logger.error("attributes({}) != special_nodes({}) - invalid_edges({}), attributes={}, special_nodes={},mrp_graph={},\n all_forward_edges={}".format(len(attributes), len(special_nodes), len(invalid_edges), attributes, special_nodes, mrp_graph.encode(), all_forward_edges))
        if len(attributes) != len(visited_edges)/2 - len(mrp_graph.edges) - len(invalid_edges):
            logger.error("attributes({}) != uni_visited_edges({}) - uni_mrp_edges({}) - invalid_edges({}),attributes = {}, invalid_edges={}, mrp_graph={}, visited_edges={}".format(len(attributes), len(visited_edges)/2, len(mrp_graph.edges), len(invalid_edges), attributes, invalid_edges, mrp_graph.encode(), visited_edges))
        return mrp_graph, s

    def getMaskAndLengths(self,batch):
        #srcBatch: len, batch, n_feature
        lengths = []

        if isinstance(batch, tuple):
            batch = batch[0]
        # batch[:,:,0] is TXT_WORD, leBatch is 2 dim, a nested list [batch_size x src_len]
        leBatch = batch[:,:,0].transpose(0,1).tolist()
        for i in range(len(leBatch)):
            # the first PAD index or the end
            l = leBatch[i].index(PAD) if PAD in  leBatch[i] else len(leBatch[i])
            lengths.append(l)
        leBatch = batch[:,:,TXT_LEMMA]
        mask = (leBatch != PAD)

        # mask : [src_len, batch]
        # leBatch: [src_len, batch]
        # lengths is a list
        return mask,leBatch,lengths


    def probToId(self, srcBatch,src_charBatch, probBatch,sourceBatch):
        #batch of id
        out = []
        assert isinstance(srcBatch, PackedSequence)

        srcBatchData = unpack(srcBatch)[0]
        mask ,leBatch,lengths= self.getMaskAndLengths(srcBatchData)
        # mask : [src_len, batch]
        # leBatch: [src_len, batch]
        # lengths is a list
        # logger.info("mask:{}, leBatch:{}, lengths:{}".format(mask, leBatch, lengths))
        # after unsqueese, [src_len, batch, 1]
        mask = mask.unsqueeze(2)
        # probBatch is tuple of packed sequence
        for i,prob in enumerate(probBatch):
            # prob is a packed squence, which is tuple
            assert isinstance(prob, tuple)
            # probData is padded tensor, [src_len, batch, n_out]
            probData = unpack(prob)[0]
            if i== AMR_LE:
                n_out = probData.size(2) # src_len x batch x n_out
                # best: best values in each row of probData in dim=2, [src_len]
                # max_indice : [src_len x batch x 1]
                best,max_indices, = probData.max(dim=2,keepdim=True)
                #logger.info("le: best : {}, max_indices :{}, prob:{}".format(best, max_indices, prob))
                assert max_indices.size(-1) == 1,("start",i, best,max_indices)
                # h: for LE, COPY for the last n_out
                # copy from lemma [src, batch x 1]
                h = (max_indices==n_out).long()*srcBatchData[:,:,TXT_LEMMA].unsqueeze(2)
                assert h.size(-1) == 1,("middle",i, best.size(),h.size(),max_indices.size(),srcBatchData[:,:,TXT_LEMMA].size())
                # max_indices, merge normal le and AMR_COPY [src, batch, 1]
                max_indices = max_indices*(max_indices<n_out).long()+h
                assert max_indices.size(-1) == 1,("middle",i, best.size(),max_indices.size())
                # mask : [src_len, batch, 1]
                max_indices = max_indices*mask.long()
                assert max_indices.size(-1) == 1,("end",i, best.size(),max_indices.size(),mask.size())
                out.append(max_indices)
            else:
                best,max_indices, = probData.max(dim=2,keepdim=True)
                #logger.info("i {} : best : {}, max_indices :{}, prob:{}".format(i, best, max_indices, prob))
                out.append(max_indices)
        l = out[0].size(2)
        for d in out:
            assert d.size(2) ==l, ([dd.size() for  dd in out],"\n",srcBatchData,probBatch,"\n",[" ".join(src[0])+"\n" for src in sourceBatch])
        # out : [src_len, batch_size, n_feature]
        # lengths: src_len for each sentence
        return torch.cat(out,2),lengths

    def probAndSourceToConcepts(self, sourceBatch,srcBatch, src_charBatch, probBatch,getsense=False):
        #batch of id
        #max_indices len,batch, n_feature
        #srcBatch  batch source
        #out batch AMRuniversal
        def id_to_high(max_le_id):
            out = []
            for id in max_le_id:
                if id < self.n_high:
                    out.append( self.dicts["amr_high_dict"].getLabel(id))
                else:
                    out.append(None)
            return out
        max_indices, lengths = self.probToId(srcBatch,src_charBatch, probBatch,sourceBatch)
        out = []
        srl_concepts = []
        dependent_mark_batch = []
        aux_triples_batch = []
        aligns = []
        for i,source in enumerate(sourceBatch):
            snt,lemma = source[0:2]
            ners = source[3]
            max_id = max_indices[:,i,:]
            cats = self.dicts["amr_category_dict"].convertToLabels(max_id[:,AMR_CAT].tolist())
            aux = self.dicts["amr_aux_dict"].convertToLabels(max_id[:,AMR_NER].tolist())
            high = id_to_high(max_id[:,AMR_LE].tolist())
      #      high = max_indices==n_out if
            if AMR_SENSE< max_id.size(1) and False:
                sense = self.dicts["amr_aux_dict"].convertToLabels(max_id[:,AMR_SENSE].tolist())
            else:
                sense = None

            amr_seq = self.rl.toAmrSeq(cats,snt,lemma,high,aux,sense,ners )
            #logger.info("snt:{}, max_id: {}, cats :{}, aux:{}, high:{}, amr_seq:{}".format(snt, max_id, cats, aux, high, amr_seq))
            # here align is for the sentence after combination.
            srl_concept,align,dependent_mark = self.fragment_to_node_converter.unpack_recategorized(amr_seq,self.rl ,getsense,eval= not self.training)

            if len(srl_concept) == 0 :
                srl_concept = [AMRUniversal("amr-unintelligible",Rule_Concept,None)]
                align = [0]
                dependent_mark = [0]
                aux_triples = []
       #         print (amr_seq,source)
       #         print ()


            srl_concepts.append(srl_concept)
            aligns.append(align)
            out.append(amr_seq)
            dependent_mark_batch.append(dependent_mark)

        return out,srl_concepts,aligns,dependent_mark_batch


    def graph_to_quadruples(self,graph):
        def add_to_quadruples(h_v,d_v,r,r_inver):
            if is_core(r):
                quadruples.append([graph.node[h_v]['value'],graph.node[d_v]['value'],r,h_v,d_v])
            else:
                quadruples.append([graph.node[d_v]['value'],graph.node[h_v]['value'],r_inver,d_v,h_v])
        quadruples = []

        for n,d in graph.nodes(True):
            if d["value"].le == BOS_WORD:
                continue
            for nearb in graph[n]:
                for key, edge_data in graph[n][nearb].items():
                    if edge_data["role"] in self.dicts["amr_rel_dict"] :
                        r = edge_data["role"]
                        quadruples.append([graph.node[n]['value'],graph.node[nearb]['value'],r,n,nearb])
                    elif edge_data["role"] == ":top-of":
                        quadruples.append([graph.node[nearb]['value'],graph.node[n]['value'],":top",nearb,n])
                    else:
                        pass

        return quadruples

    def graph_to_concepts(self,graph):
        concepts = []

        for n,d in graph.nodes(True):
            if d["value"].le == BOS_WORD:
                continue
            add = True
            for nearb in graph[n]:
                all_wiki = all([":wiki-of" in edge_data["role"] for key, edge_data in graph[n][nearb].items()])
                if all_wiki:
                    add = False
                    break
            if add:
                concepts.append(d["value"])
        return concepts

    def graph_to_concepts_batches(self,graphs):
        return [self.graph_to_concepts(graph) for graph in graphs]

    def relProbAndConToGraph(self,srl_batch, sourceBatch, srl_prob,roots,appended,get_sense=False,set_wiki=False,normalizeMod=False):
        #batch of id
        #max_indices len,batch, n_feature
        #srcBatch  batch source
        #out batch AMRuniversal
        amr_rel_dict = self.dicts["amr_rel_dict"]
        def get_uni_var(concepts,id):
            assert id < len(concepts),(id,concepts)
            uni = concepts[id]
            if  uni.le in [ "i" ,"it","you","they","he","she"] and uni.cat == Rule_Concept:
                return uni,Var(uni.le )
            le = uni.le
            if uni.cat != Rule_String:
                uni.le = uni.le.strip("/").strip(":")
                if ":" in uni.le or "/" in uni.le:
                    uni.cat = Rule_String
            if uni.le == "":
                return uni,Var(le+ str(id))
            return uni,Var(uni.le[0]+ str(id))

        def create_connected_graph(role_scores,concepts,root_id,dependent,aligns, normalizeMod=True):
            #role_scores: amr x amr x rel
            graph = nx.MultiDiGraph()
            n = len(concepts)
            role_scores = role_scores.view(n,n,-1)
            max_non_score, max_non_score_id= role_scores[:,:,1:].max(-1)
            max_non_score_id = max_non_score_id +1
            non_score = role_scores[:,:,0]
            active_cost =   non_score - max_non_score  #so lowest cost edge gets to active first
            candidates = []
            # add all nodes
            for h_id in range(n):
                h,h_v = get_uni_var(concepts,h_id)
                # here align is a list to the token index in our tokenization
                graph.add_node(h_v, value=h, align=aligns[h_id],gold=True,dep = dependent[h_id])

            constant_links = {}
            normal_edged_links = {}
            # add all pairs of edges
            for h_id in range(n):
                for d_id in range(n):
                    if h_id != d_id:
                        r = amr_rel_dict.getLabel(max_non_score_id[h_id,d_id].item())
                        # relations in amr_rel_dict are forward argx, opx, sntx top
                        # and all backward relations.
                        # normalize mod should already be done in amr_rel_dict
                        if normalizeMod and to_be_normalize_mod(r):
                            r = get_normalize_mod(r)
                        # we should normalize it here when connecting, make sure to not consider the same relation twices
                        r_inver = get_inversed_edge(r)
                        h,h_v = get_uni_var(concepts,h_id)
                        d,d_v = get_uni_var(concepts,d_id)
                        if  (concepts[h_id].is_constant() or concepts[d_id].is_constant() ):
                            if concepts[h_id].is_constant() and concepts[d_id].is_constant() :
                                continue
                            elif concepts[h_id].is_constant():
                                constant_links.setdefault(h_v,[]).append((active_cost[h_id,d_id],d_v,r,r_inver))
                            else:
                                constant_links.setdefault(d_v,[]).append((active_cost[h_id,d_id],h_v,r_inver,r))
                        elif active_cost[h_id,d_id] < 0:
                            if r in [":name-of" ,":ARG0"] and concepts[d_id].le in ["person"]:
                                # always adding two direction to support directed path for connectivity
                                graph.add_edge(h_v, d_v, key=r, role=r)
                                graph.add_edge(d_v, h_v, key=r_inver, role=r_inver)
                            else:
                         #       if concepts[h_id].le == "name" and r != ":name-of":
                          #          r = ":name-of"
                                normal_edged_links.setdefault((h_v,r),[]).append((active_cost[h_id,d_id],d_v,r_inver))
                        else:
                            candidates.append((active_cost[h_id,d_id],(h_v,d_v,r,r_inver)))

            max_edge_per_node = 1 if not self.training else 100
            for h_v,r in normal_edged_links:
                sorted_list = sorted(normal_edged_links[h_v,r],key = lambda j:j[0])
                for _,d_v,r_inver in sorted_list[:max_edge_per_node]:
               #     if graph.has_edge(h_v, d_v):
               #         continue
                    graph.add_edge(h_v, d_v, key=r, role=r)
                    graph.add_edge(d_v, h_v, key=r_inver, role=r_inver)
                for cost,d_v,r_inver in sorted_list[max_edge_per_node:]:  #remaining
                    candidates.append((cost,(h_v,d_v,r,r_inver)))


            for h_v in constant_links:
                _,d_v,r,r_inver = sorted(constant_links[h_v],key = lambda j:j[0])[0]
                graph.add_edge(h_v, d_v, key=r, role=r)
                graph.add_edge(d_v, h_v, key=r_inver, role=r_inver)

            candidates = sorted(candidates,key = lambda j:j[0])

            for _,(h_v,d_v,r,r_inver ) in candidates:
                if  nx.is_strongly_connected(graph):
                    break
                if not nx.has_path(graph,h_v,d_v):
                    graph.add_edge(h_v, d_v, key=r, role=r,force_connect=True)
                    graph.add_edge(d_v, h_v, key=r_inver, role=r_inver,force_connect=True)

            _,root_v  = get_uni_var(concepts,root_id)
            h_v = BOS_WORD
            root_symbol = AMRUniversal(BOS_WORD,BOS_WORD,NULL_WORD)
            graph.add_node(h_v, value=root_symbol, align=-1,gold=True,dep=1)
            graph.add_edge(h_v, root_v, key=":top", role=":top")
            graph.add_edge(root_v, h_v, key=":top-of", role=":top-of")

            if get_sense:
                for n,d in graph.nodes(True):
                    if "value" not in d:
                        print (n,d, graph[n],constant_links,graph.nodes,graph.edges)
                    le,cat,sense = d["value"].le,d["value"].cat,d["value"].sense
                    if cat == Rule_Frame and sense == "":
                        sense = self.fragment_to_node_converter.get_senses(le)
                        d["value"] = AMRUniversal(le,cat,sense)

            if not self.training:
                # when not training, it didn't do contratact
                if not nx.is_strongly_connected(graph):
                    logger.warn("not connected before contraction: %s, %s, %s, %s, %s, %s, %s, %s",self.graph_to_quadruples(graph),graph_to_amr(graph), candidates, constant_links, graph.nodes(), graph.edges(), normal_edged_links, concepts)
                graph = contract_graph(graph)

            if set_wiki:
                list = [[n,d]for n,d in graph.nodes(True)]
                for n,d in list:
                    if d["value"].le == "name":
                        names = []
                        head = None
                        for nearb in graph[n]:
                            for key, edge_data in graph[n][nearb].items():
                                r = edge_data["role"]
                                if ":op" in r and "-of" not in r and  int(edge_data["role"][3:]) not in names:
                                    names.append([graph.node[nearb]["value"], int(edge_data["role"][3:])])
                                if r == ":name-of":
                                    wikied = False
                                    for nearbb in graph[nearb]:
                                        for _, edge_data2 in graph[nearb][nearbb].items():
                                            r2 = edge_data2["role"]
                                            if r2 == ":wiki":
                                                wikied = True
                                                break
                                    if not wikied:
                                        head = nearb
                        if head:
                            names = tuple([t[0] for t in sorted(names,key = lambda t: t[1])])
                            wiki = self.fragment_to_node_converter.get_wiki(names)
                        #    print (wiki)
                            wiki_v = Var(wiki.le+n._name )
                            graph.add_node(wiki_v, value=wiki, align=d["align"],gold=True,dep=2) #second order  dependency
                            graph.add_edge(head, wiki_v, key=":wiki", role=":wiki")
                            graph.add_edge(wiki_v, head, key=":wiki-of",role=":wiki-of")

            if not nx.is_strongly_connected(graph):
                logger.warn("not connected after contraction: %s, %s, %s, %s, %s, %s, %s, %s",self.graph_to_quadruples(graph),graph_to_amr(graph), candidates, constant_links, graph.nodes(), graph.edges(), normal_edged_links, concepts)
            return graph,self.graph_to_quadruples(graph)

        graphs = []
        quadruple_batch = []
        score_batch = myunpack(*srl_prob) #list of (h x d)


        depedent_mark_batch = appended[0]
        #here aligns_batch is how final expaned node aligned to the categorized node
        aligns_batch = appended[1]
        for i,(role_scores,concepts,roots_score,dependent_mark,aligns) in enumerate(zip(score_batch,srl_batch,roots,depedent_mark_batch,aligns_batch)):
            root_s,root_id = roots_score.max(0)
            assert roots_score.size(0) == len(concepts),(concepts,roots_score)
            # in pytorch 0.4.0 to https://pytorch.org/docs/stable/tensors.html#torch.Tensor.tolist
            # tensor.tolist can be a int, when root_id is a scalar
            root_id = root_id.item()
            assert root_id < len(concepts),(concepts,roots_score)

            g,quadruples = create_connected_graph(role_scores,concepts,root_id,dependent_mark,aligns, normalizeMod=normalizeMod)
            graphs.append(g)
            quadruple_batch.append(quadruples)

        return graphs,quadruple_batch

    def nodes_jamr(self,graph):
        s = []
        for n,d in graph.nodes(True):
            if d["value"].le == BOS_WORD:
                continue
            a = d["align"].item()
            assert isinstance(a,int),(n,a,graph.nodes(True))
            if a > -1:
                s.append("# ::node\t"+n._name+"\t"+d["value"].gold_str()+"\t"+ str(a)+"-"+str(a+1)+"\n") #+"\t"+str(d["dep"])
        return "".join(s)

# ::edge	border-01	ARG2	country	0.1.0	0.1.0.2
    def edges_jamr(self,graph,dep_only=False):
        s = []
        def cannonical(r):
            return  "-of" in r and not is_core(r) or  "-of"  not in r and  is_core(r)
        for n,d in graph.nodes(True):
            if d["value"].le == BOS_WORD:
                continue
            if dep_only:
                for nearb in graph[n]:
                    if graph.node[nearb]["dep"] > d["dep"] and  graph.node[nearb]["value"].le != BOS_WORD:
                        head = graph.node[n]["value"].gold_str()
                        dep = graph.node[nearb]["value"].gold_str()
                        for key, edge_data in graph[n][nearb].items():
                            r = edge_data["role"]
                            assert isinstance(n,Var),(n,graph.nodes(True))
                            assert isinstance(nearb,Var),(nearb,graph.nodes(True))
                            s.append("# ::edge\t"+head+"\t"+ r+"\t"+dep+"\t"+n._name+"\t"+nearb._name+"\t"+"\n")
            else:
                for nearb in graph[n]:
                    for key, edge_data in graph[n][nearb].items():
                        r = edge_data["role"]
                        if cannonical(r):
                            head = graph.node[n]["value"].gold_str()
                            dep = graph.node[nearb]["value"].gold_str()
                            for key, edge_data in graph[n][nearb].items():
                                r = edge_data["role"]
                                assert isinstance(n,Var),(n,graph.nodes(True))
                                assert isinstance(nearb,Var),(nearb,graph.nodes(True))
                                s.append("# ::edge\t"+head+"\t"+ r+"\t"+dep+"\t"+n._name+"\t"+nearb._name+"\t"+"\n")
        return "".join(s)

    
