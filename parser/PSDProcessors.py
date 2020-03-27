#!/usr/bin/env python3.6
# coding=utf-8
'''

PSDParser for producing psd graph from raw text
PSDDecoder for decoding deep learning model output into actual PSD nodes and graph
PSDInputPreprocessor for extract features based on stanford corenlp
@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30

# This PSDProcessors are used as a luanch for preprocessing the input raw sentence and parse it
@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-01
'''
from parser.Dict import seq_to_id
from utility.constants import core_nlp_url
import networkx as nx
from utility.mtool.graph import *
import random
import string
import copy

from src import *
from utility.psd_utils.PSDGraph import PSDGraph
from utility.psd_utils.PSDStringCopyRules import *
from utility.psd_utils.PSDReCategorization import *
from parser.modules.helper_module import myunpack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence
from parser.DataIterator import DataIterator,rel_to_batch
from pycorenlp import StanfordCoreNLP
import logging

logger = logging.getLogger("mrp.parser.PSDProcessors")

class PSDInputPreprocessor(object):
    """
    A feature extractor for psd input
    """
    def __init__(self, opt, url = core_nlp_url, joints_file=psd_mwe_file):
        self.nlp = StanfordCoreNLP(url)
        self.opt = opt
        self.joints_map = copy.deepcopy(g_vallex_reader.joints_map)
        self.load_extra_joints(joints_file)
        logger.info("joint_map loaded: {}".format(self.joints_map))

    # load extra joints
    def load_extra_joints(self, joints_file="psd.joints"):
        with open(joints_file,'r') as fin:
            for line in fin:
                splits = line.rstrip('\n').split(" ")
                if len(splits) > 1 and splits[1]!="":
                    compounds = splits+[MWE_END]
                    past = ""
                    for w in compounds:
                        self.joints_map.setdefault(past[:-1],[]).append(w)
                        past = past + w + "+"

    def featureExtract(self,src_text,whiteSpace=False):
        """
        Using stanford nlp url to extract features from psd input text
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
            data["lem"].append(snt_tok['lemma'])
            data["pos"].append(snt_tok['pos'])
            # first add anchors as a dictionary here.
            data["anchors"].append([{'from': snt_tok['characterOffsetBegin'], 'to': snt_tok['characterOffsetEnd']}])
     #   if whiteSpace is False:
     #       return self.featureExtract(" ".join(data["tok"]),True)
        asserting_equal_length(data)
        return data

    def preprocess(self,src_text, whiteSpace=False, token_combine=False):
        data = self.featureExtract(src_text, whiteSpace)
        data = self.annotate_mwe(data)
        return data

    def annotate_mwe(self, data):
        acc = ""
        mwe = ['O']*len(data['lem'])
        start = 0
        skip = False
        i = 0
        for i, le in enumerate(data["lem"]):
            if skip:
                skip = False
            elif len(acc) > 0 and le in self.joints_map.get(acc, []) :
                acc = acc +"_"+le
                if MWE_END in self.joints_map.get(acc,[]):
                    for j in range(start, i+1):
                        mwe[j] = acc
            elif len(acc) > 0 and le == "-" and i < len(data["lem"])-1 \
                and data["lem"][i+1] in self.joints_map.get(acc, []):
                acc = acc +"_"+data["lem"][i+1]
                if MWE_END in self.joints_map.get(acc,[]):
                    for j in range(start, i+2):
                        mwe[j] = acc
                skip = True
            else:
                acc = le
                start = i

        data["mwe"] = mwe
        return data

class PSDParser(object):
    """
    PSD Parser class
    """
    def __init__(self, opt,dicts):
        self.decoder = PSDDecoder(opt,dicts,"psd")
        self.frame = "psd"
        self.model, _, _ = load_old_model(dicts,opt,True)[0]
        self.opt = opt
        self.feature_extractor = PSDInputPreprocessor(opt,core_nlp_url)
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
            data_iterator = BertDataIterator([],self.opt,self.dicts["psd_rel_dict"],all_data=all_data)
            order,idsBatch,srcBatch, src_charBatch, sourceBatch,srcBertBatch = data_iterator[0]
        else:
            data_iterator = DataIterator([],self.opt,self.dicts["psd_rel_dict"],all_data=all_data)
            order,idsBatch,srcBatch, src_charBatch, sourceBatch = data_iterator[0]
            srcBertBatch = None
        return order,idsBatch,srcBatch, src_charBatch, sourceBatch,srcBertBatch,data_iterator

    def parse_batch_preprocessed_data(self, preprocessed_input_examplesi, set_wiki, normalize_mod):
        """
        only mrp outputs, no txt outputs
        """
        order,idsBatch,srcBatch, src_charBatch, sourceBatch,srcBertBatch,srcBertIndexBatch,data_iterator = self.feature_to_torch(preprocessed_input_examples)

        probBatch, src_enc = self.model(srcBatch, src_charBatch, rel=False, bertBatch=srcBertBatch, bertIndexBatch=srcBestIndexBatch)

        concepts_pred_seq,concept_batches,aligns_raw,dependent_mark_batch = self.decoder.probAndSourceToConcepts(sourceBatch,srcBatch, src_charBatch, probBatch,getsense = True)

        concepts_pred_seq = [ [(uni.cat,uni.le,uni.aux,uni.sense,uni)  for uni in seq ] for  seq in concepts_pred_seq ]


        rel_batch,aligns = rel_to_batch(concept_batches,aligns_raw,data_iterator,self.dicts)
        rel_prob,roots = self.model((rel_batch,srcBatch, src_charBatch, src_enc,aligns), rel=True, bertBatch=srcBertBatch, bertIndexBatch=srcBertIndexBatch)
        graphs,rel_triples  =  self.decoder.relProbAndConToGraph(concept_batches,rel_prob,roots,(dependent_mark_batch,aligns),True,set_wiki,normalizeMod=normalize_mod)
        batch_out = [0]*len(graphs)

        batch_mrp_graphs = [0]*len(graphs)
        for i,data in enumerate(zip(idsBatch, sourceBatch,concepts_pred_seq,concept_batches,rel_triples,graphs)):
            example_id, source,amr_pred,concept, rel_triple,graph= data
            mrp_graph, predicated_graph = self.decoder.graph_to_mrpGraph(example_id, graph, normalizeMod = opt.normalize_mod, flavor=0, framework="psd", sentence=" ".join(source[0]))
            batch_mrp_graphs[order[i]] = mrp_graph
        return batch_mrp_graphs, batch_out

    def parse_one_preprocessed_data(self, preprocessed_input_example, set_wiki, normalize_mod):
        return self.parse_batch_preprocessed_data([preprocessed_input_example], set_wiki, normalize_mod)

    def parse_batch(self,src_text_batch, set_wiki, normalize_mod):
        all_data =[ self.feature_extractor.preprocess(src_text) for src_text in src_text_batch ]
        return self.parse_batch_preprocessed_data(all_data, set_wiki, normalize_mod)

    def parse_one(self,src_text, set_wiki, normalize_mod):
        return self.parse_batch([src_text], set_wiki, normalize_mod)

#decoder requires copying dictionary, and recategorization system
#full_tricks evoke some deterministic post processing, which might be expensive to compute
class PSDDecoder(object):
    def __init__(self, opt,dicts):
        """
        init the model for PSD Decoder in pytorch
        Especially, it relys on the dicts/vocabularies for tokens and labels.
        """
        self.opt = opt
        self.dicts = dicts
        self.frame = "psd"
        self.n_high = len(dicts["psd_high_dict"])
        self.rl = PSDRules()
        self.rl.load(opt.build_folder+"dicts/psd_rule_f")
        self.fragment_to_node_converter = PSDReCategorizor(from_file=True,path=opt.build_folder+"dicts/psd_recategorization",training=False)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @staticmethod
    def graph_to_mrpGraph(id, G, normalizeMod=False, flavor = 0, framework = "psd", sentence = None):
        mrp_graph = Graph(id, flavor = flavor, framework=framework)
        top = BOS_WORD
        root = list(G[top].keys())[0]
        root_uni = G.node[root]["value"]
        # it is better to add node by the order of anchors
        nodes = [x for x in G.node if x!=top]
        nodes.sort(key=lambda x: G.node[x]["anchors"][0]["from"])
        for n in nodes:
            is_top = False
            if n == root:
                is_top = True
            uni = G.node[n]["value"]
            node_label = uni.le
            pos = uni.pos
            anchors = G.node[n]["anchors"]

            frame = uni.get_frame()
            if frame:
                node = mrp_graph.add_node(label=node_label, properties=["pos","frame"], values=[pos,frame], anchors = anchors,top=is_top)
            else:
                node = mrp_graph.add_node(label=node_label, properties=["pos"], values=[pos], anchors = anchors,top=is_top)

            G.node[n]["mrp_node"]=node

        for e in G.edges:
            # only add forward relation
            h = e[0]
            d = e[1]
            role = G.edges[e]["role"]
            #if "reduced" in G.edges[e]:
            #    continue
            if PSDGraph.is_inversed_edge(role):
                continue
            elif role == ":top":
                assert "mrp_node" in G.node[e[1]], "mrp_node {} not created, but visited and not constant".format(G.node[h])
                G.node[e[1]]["mrp_node"].is_top = True
            else:
                assert "mrp_node" in G.node[h], "mrp_node {} not created, but visited and not constant".format(G.node[h])
                assert "mrp_node" in G.node[d], "mrp_node {} not created, but visited and not constant".format(G.node[d])
                src = G.node[h]["mrp_node"].id
                tgt = G.node[d]["mrp_node"].id
                edge_label = role[1:]
                mrp_graph.add_edge(src, tgt, edge_label)
        # when root is constant, not use attributes
        return mrp_graph, None

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


    def probToId(self, srcBatch, src_charBatch, probBatch,sourceBatch):
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
            if i== PSD_LE:
                n_out = probData.size(2) # src_len x batch x n_out
                high_index = n_out - 2
                copy_lemma_index = n_out - 1
                # best: best values in each row of probData in dim=2, [src_len]
                # max_indice : [src_len x batch x 1]
                best,max_indices, = probData.max(dim=2,keepdim=True)
                # logger.info("max_indices :{}".format(max_indices))
                assert max_indices.size(-1) == 1,("start",i, best,max_indices)
                # h: for LE, COPY for the last n_out
                # copy from lemma [src, batch x 1]
                h = (max_indices==copy_lemma_index).long()*srcBatchData[:,:,TXT_LEMMA].unsqueeze(2)
                assert h.size(-1) == 1,("middle",i, best.size(),h.size(),max_indices.size(),srcBatchData[:,:,TXT_LEMMA].size())
                # max_indices, merge normal le and AMR_COPY [src, batch, 1]
                # max_indices = max_indices*(max_indices <= high_position).long()+ h + h0
                max_indices = max_indices*(max_indices <= high_index).long()+ h
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
                    out.append(self.dicts["psd_high_dict"].getLabel(id))
                else:
                    # this part is not differentiable, we leave it for determistic mapping
                    out.append(None)
            return out

        # for PSD, sense is used in relation part, and we trust the classfied sense, not manually get sense from rules
        max_indices, lengths = self.probToId(srcBatch, src_charBatch, probBatch,sourceBatch)
        out = []
        srl_concepts = []
        dependent_mark_batch = []
        aux_triples_batch = []
        aligns = []
        for i,source in enumerate(sourceBatch):
            snt = source[TOK_IND_SOURCE_BATCH]
            lemma = source[LEM_IND_SOURCE_BATCH]
            ners = source[NER_IND_SOURCE_BATCH]
            mwes= source[MWE_IND_SOURCE_BATCH]
            max_id = max_indices[:,i,:]
            high = id_to_high(max_id[:,PSD_LE].tolist())
            # when the le is beyond the high_dict, fix that in toPSDSeq in rules
            pos = self.dicts["psd_target_pos_dict"].convertToLabels(max_id[:,PSD_POS].tolist())
            #lemmas = self.dicts["lemma_dict"].convertToLabels(max_id[:,PSD_LE].tolist())
            # here always no predicate the sense
            # here, we only add sense for psd now.
            # for psd, we use the predicated sense
            if PSD_SENSE< max_id.size(1):
                sense = self.dicts["psd_sense_dict"].convertToLabels(max_id[:,PSD_SENSE].tolist())
            else:
                sense = None

            psd_seq = self.rl.toPSDSeq(pos, snt,lemma,high, sense,ners, mwes)
            # TODO: show concepts, align

            # TODO: here adding gold concepts here.
            srl_concept,align,dependent_mark = self.fragment_to_node_converter.unpack_recategorized(psd_seq, self.rl ,getsense,eval= not self.training)
            if len(srl_concept) == 0:
                # TODO: here force it not empty, but nodes may be empty sometimes
                srl_concept = [ PSDUniversal.NULL_PSDUniversal()]
                align = [0]
                dependent_mark = [0]
                aux_triples = []

            srl_concepts.append(srl_concept)
            aligns.append(align)
            # psd is some predict senquence,srl_concept, it is some concept after expandsion.
            out.append(psd_seq)
            dependent_mark_batch.append(dependent_mark)

        return out,srl_concepts,aligns,dependent_mark_batch

    def graph_to_quadruples(self,graph):
        """
        from networkx.MultiDiGraph to quadruples
        """
        quadruples = []

        for n,d in graph.nodes(True):
            if d["value"].le == BOS_WORD:
                # top node
                continue
            for nearb in graph[n]:
                for key, edge_data in graph[n][nearb].items():
                    #if edge_data["role"] in self.dicts["psd_rel_dict"] or edge_data["role"].replace("-arg","") in self.dicts["psd_rel_dict"]:
                    if edge_data["role"] in self.dicts["psd_rel_dict"]:
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
            concepts.append(d["value"])
        return concepts

    def graph_to_concepts_batches(self,graphs):
        return [self.graph_to_concepts(graph) for graph in graphs]

    def relProbAndConToGraph(self,srl_batch, sourceBatch, srl_prob,roots,appended,get_sense=False,set_wiki=False,normalizeMod=False):
        """
        For PSD, there is normalizeMod
        """
        # TODO: graph connected graph given relation prob
        #batch of id
        #max_indices len,batch, n_feature
        #srcBatch  batch source
        #out batch AMRuniversal
        rel_dict = self.dicts["psd_rel_dict"]
        def get_uni_var(concepts,id):
            """
            for psd, id is also not important, just use the id as variable value
            """
            return concepts[id],PSDVar(str(id))

        def create_connected_graph(role_scores,concepts,root_id,dependent,aligns,source):
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
                graph.add_node(h_v, value=h, align=aligns[h_id],gold=True,dep = dependent[h_id])

            constant_links = {}
            normal_edged_links = {}
            # add all pairs of edges
            for h_id in range(n):
                for d_id in range(n):
                    if h_id != d_id:
                        h,h_v = get_uni_var(concepts,h_id)
                        d,d_v = get_uni_var(concepts,d_id)
                        r = rel_dict.getLabel(max_non_score_id[h_id,d_id].item())
                        #r_inver = PSDGraph.get_inversed_edge(r)
                        # relations in rel_dict are forward argx, opx, sntx top
                        # and all backward relations.
                        # normalize mod should already be done in rel_dict
                        # we should normalize it here when connecting, make sure to not consider the same relation twices
                        # make sure -arg is fixed at last
                        if not PSDGraph.is_must_arg_functor(r) and PSDGraph.is_arg_functor(r):
                            # we should add -arg if r is in args list
                            if PSDGraph.is_inversed_edge(r):
                                r_inver = PSDGraph.get_inversed_edge(r)
                                # r_inver here is forward
                                if PSDGraph.check_oblig_args_in_vallex(r_inver, d):
                                    r_inver = r_inver +"-arg"
                                    r = PSDGraph.get_inversed_edge(r_inver)
                                else:
                                    pass
                            else:
                                # if forward, not considering -arg, because forward is top, -args or .members
                                # r is forward, then r_inver is backward
                                r_inver = PSDGraph.get_inversed_edge(r)
                        else:
                            # if not arg functor, not considering -arg
                            r_inver = PSDGraph.get_inversed_edge(r)
                        # TODO: for mwe, compound ner
                        if active_cost[h_id,d_id] < 0:
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

            candidates = sorted(candidates,key = lambda j:j[0])

            for _,(h_v,d_v,r,r_inver ) in candidates:
                if  nx.is_strongly_connected(graph):
                    break
                if not nx.has_path(graph,h_v,d_v):
                    graph.add_edge(h_v, d_v, key=r, role=r,force_connect=True)
                    graph.add_edge(d_v, h_v, key=r_inver, role=r_inver,force_connect=True)

            _,root_v  = get_uni_var(concepts,root_id)
            h_v = BOS_WORD
            root_symbol = PSDUniversal.TOP_PSDUniversal()
            graph.add_node(h_v, value=root_symbol, align=-1,gold=True,dep=1)
            graph.add_edge(h_v, root_v, key=":top", role=":top")
            graph.add_edge(root_v, h_v, key=":top-of", role=":top-of")

            for n,d in graph.nodes(True):
                le,pos,sense= d["value"].le,d["value"].pos,d["value"].sense
                # here align is the token index list
                # last saving for not assigned sense. mainly for frames check
                if get_sense:
                    sense = self.fragment_to_node_converter.fix_sense(pos, le, sense)

                anchors = []
                # convert token ids into character offset, with input source data
                tok_index = d["align"]
                if tok_index >= 0 and tok_index < len(source[ANCHOR_IND_SOURCE_BATCH]):
                    anchors.extend(source[ANCHOR_IND_SOURCE_BATCH][tok_index])

                d["value"] = PSDUniversal(pos,le,sense,anchors)
                d["anchors"] = anchors

            if not nx.is_strongly_connected(graph):
                logger.warn("not connected after contraction: %s, %s, %s, %s, %s, %s, %s, %s",self.graph_to_quadruples(graph),graph_to_amr(graph), candidates, constant_links, graph.nodes(), graph.edges(), normal_edged_links, concepts)
            return graph,self.graph_to_quadruples(graph)

        graphs = []
        quadruple_batch = []
        score_batch = myunpack(*srl_prob) #list of (h x d)


        depedent_mark_batch = appended[0]
        aligns_batch = appended[1]
        for i,(role_scores,concepts,roots_score,dependent_mark,aligns,source) in enumerate(zip(score_batch,srl_batch,roots,depedent_mark_batch,aligns_batch,sourceBatch)):
            root_s,root_id = roots_score.max(0)
            assert roots_score.size(0) == len(concepts),(concepts,roots_score)
            # in pytorch 0.4.0 to https://pytorch.org/docs/stable/tensors.html#torch.Tensor.tolist
            # tensor.tolist can be a int, when root_id is a scalar
            root_id = root_id.item()
            assert root_id < len(concepts),(concepts,roots_score)

            g,quadruples = create_connected_graph(role_scores,concepts,root_id,dependent_mark,aligns,source)
            graphs.append(g)
            quadruple_batch.append(quadruples)

        return graphs,quadruple_batch
