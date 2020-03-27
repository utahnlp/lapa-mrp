#!/usr/bin/env python3.6
# coding=utf-8
'''

UCCAParser for producing ucca graph from raw text
UCCADecoder for decoding deep learning model output into actual UCCA nodes and graph
UCCAInputPreprocessor for extract features based on stanford corenlp
@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30

# This UCCAProcessors are used as a luanch for preprocessing the input raw sentence and parse it
@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-01
'''
from parser.Dict import seq_to_id
from utility.constants import core_nlp_url
import networkx as nx
from utility.mtool.graph import *
import random
import string
import sys

from src import *
from utility.ucca_utils.UCCAGraph import UCCAGraph
from utility.ucca_utils.UCCAStringCopyRules import *
from utility.ucca_utils.UCCAReCategorization import *
from parser.modules.helper_module import myunpack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import PackedSequence
from parser.DataIterator import DataIterator,rel_to_batch
from pycorenlp import StanfordCoreNLP
import copy
import logging

logger = logging.getLogger("mrp.UCCAProcessors")

class UCCAInputPreprocessor(object):
    """
    A feature extractor for ucca input
    """
    def __init__(self, opt, url = core_nlp_url):
        self.nlp = StanfordCoreNLP(url)
        self.opt = opt

    def featureExtract(self,src_text,whiteSpace=True):
        """
        Using stanford nlp url to extract features from ucca input text
        whiteSpace means only split workds when there is a whitespace, it can be used to keep existed tokenization
        """
        data = {}
        output = self.nlp.annotate(src_text.strip(), properties={
        'annotators': "tokenize,ssplit,pos,lemma,ner",
        "tokenize.options":"splitHyphenated=false",
        #"tokenize.whitespace": whiteSpace,
        #"tokenize.language": "Whitespace" if whiteSpace else "English",
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

        data['mwe'] = ['O'] * len(data["tok"])
     #   if whiteSpace is False:
     #       return self.featureExtract(" ".join(data["tok"]),True)
        asserting_equal_length(data)
        return data

    def preprocess(self,src_text, whiteSpace=False, token_combine = False):
        """
        used for a pipeline for parsing
        """
        data = self.featureExtract(src_text, whiteSpace)
        return data

class UCCAParser(object):
    """
    UCCA Parser class
    """
    def __init__(self, opt,dicts):
        self.decoder = UCCADecoder(opt,dicts,"ucca")
        self.frame = "ucca"
        self.model, _, _ = load_old_model(dicts,opt,True)[0]
        self.opt = opt
        self.feature_extractor = UCCAInputPreprocessor(opt,core_nlp_url)
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
            data_iterator = BertDataIterator([],self.opt,self.dicts["ucca_rel_dict"],all_data=all_data)
            order,idsBatch,srcBatch, src_charBatch, sourceBatch,srcBertBatch = data_iterator[0]
        else:
            data_iterator = DataIterator([],self.opt,self.dicts["ucca_rel_dict"],all_data=all_data)
            order,idsBatch,srcBatch,src_charBatch, sourceBatch = data_iterator[0]
            srcBertBatch = None
        return order,idsBatch,srcBatch,src_charBatch, sourceBatch,srcBertBatch,data_iterator

    def parse_batch_preprocessed_data(self, preprocessed_input_examplesi, set_wiki, normalize_mod):
        """
        only mrp outputs, no txt outputs
        """
        order,idsBatch,srcBatch, src_charBatch,  sourceBatch,srcBertBatch,data_iterator = self.feature_to_torch(preprocessed_input_examples)

        probBatch, src_enc = self.model(srcBatch,src_charBatch, rel=False, bertBatch=srcBertBatch, bertIndexBatch=srcBertIndexBatch)

        concepts_pred_seq,concept_batches,aligns_raw,dependent_mark_batch = self.decoder.probAndSourceToConcepts(sourceBatch,srcBatch, src_charBatch, probBatch,getsense = True)

        concepts_pred_seq = [ [(uni.cat,uni.le,uni.aux,uni.sense,uni)  for uni in seq ] for  seq in concepts_pred_seq ]


        rel_batch,aligns = rel_to_batch(concept_batches,aligns_raw,data_iterator,self.dicts)
        rel_prob,roots = self.model((rel_batch,srcBatch, src_charBatch, src_enc,aligns), rel=True, bertBatch=srcBertBatch, bertIndexBatch=srcBertIndexBatch)
        graphs,rel_triples  =  self.decoderrelProbAndConToGraph(concept_batches,rel_prob,roots,(dependent_mark_batch,aligns),True,set_wiki,normalizeMod=normalize_mod)
        batch_out = [0]*len(graphs)

        batch_mrp_graphs = [0]*len(graphs)
        for i,data in enumerate(zip(idsBatch, sourceBatch,concepts_pred_seq,concept_batches,rel_triples,graphs)):
            example_id, source,amr_pred,concept, rel_triple,graph= data
            mrp_graph, catd_graph = self.decoder.graph_to_mrpGraph(example_id, graph, normalizeMod = opt.normalize_mod, flavor=0, framework="ucca", sentence=" ".join(source[0]))
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
class UCCADecoder(object):
    def __init__(self, opt,dicts):
        """
        init the model for UCCA Decoder in pytorch
        Especially, it relys on the dicts/vocabularies for tokens and labels.
        """
        self.opt = opt
        self.dicts = dicts
        self.frame = "ucca"
        self.n_high = len(dicts["ucca_high_dict"])
        self.n_le_high = len(dicts["ucca_high_le_dict"])
        self.rl = UCCARules()
        self.rl.load(opt.build_folder+"dicts/ucca_rule_f")
        self.fragment_to_node_converter = UCCAReCategorizor(from_file=True,path=opt.build_folder+"dicts/ucca_recategorization",training=False)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    @staticmethod
    def graph_to_mrpGraph(id, G, normalizeMod=False, flavor = 2, framework = "ucca", sentence = None):
        # do contraed for tok_tag first
        H = G.copy()
        for h, d, key, edge_data in G.edges(keys=True, data=True):
            # key will be changed after contraining, for example. key is no more equal to role
            if UCCAGraph.is_tok_edge(edge_data["role"]) and not UCCAGraph.is_inversed_edge(edge_data["rot"]):
                # d will be kept
                H = nx.contracted_nodes(H, d,h, self_loops=False)
        # do contraed for ME _tag first
        M = H.copy()
        for h, d, key, edge_data in H.edges(keys=True, data=True):
            if edge_data["role"] == ME_TAG and h in M.nodes:
                # h will be kept, words are contracted into its parent
                anchors = M.node[h]['anchors'] if M.node[h]['anchors'] else []
                align = M.node[h]['align'] if M.node[h]['align'] else []
                d_anchors = H.node[d]['anchors'] if H.node[d]['anchors'] else []
                for anchor in d_anchors:
                    in_h = False
                    for anchor_h in anchors:
                        if anchor['from'] == anchor_h['from'] and anchor['to'] == anchor_h['to']:
                            in_h = True
                    if not in_h:
                        anchors.append(anchor)
                for a in H.node[d]['align']:
                    if a not in align:
                        align.append(a)
                if h in M.nodes() and d in M.nodes():
                    M = nx.contracted_nodes(M, h, d, self_loops=False)
                    new_x_uni = UCCAUniversal(LEAF_NODE, anchors)
                    M.node[h]['value'] = new_x_uni
                    M.node[h]['anchors'] = anchors
                    M.node[h]['align'] = sorted(align)

        # logger.error("graph2mrp after contracting: for {}\n graph_nodes:{}\n  graph_edges:{}\n\n".format(id, M.nodes.data(), M.edges.data()))

        mrp_graph = Graph(id, flavor = flavor, framework=framework)
        if sentence:
            mrp_graph.add_input(sentence)
        top = BOS_WORD
        root = list(M[top].keys())[0]
        #try:
        #    root_uni = G.node[root]["value"]
        #except:
        #    logger.error("root uni: {},root={},  {}".format(G[top].keys(),root, G.node[root]))

        # it is better to add node by the order of anchors
        # for UCCA, we add top also
        nodes = [x for x in M.node if x != top]
        nodes.sort(key=lambda x: min([a['from'] for a in M.node[x]["anchors"]])*100 + max([a['to'] for a in M.node[x]['anchors']]) if M.node[x]["anchors"]!=None and len(M.node[x]["anchors"]) != 0 else sys.maxsize)
        for i, n in enumerate(nodes):
            is_top = False
            if n == root:
                is_top = True
            try:
                uni = M.node[n]["value"]
            except:
                logger.error("n:{}, node :{}".format(n, M.node[n]))

            if isinstance(uni, UCCAUniversal):
                if uni.ntype == INTERNAL_NODE or uni.ntype == TOP_NODE:
                    node = mrp_graph.add_node(id = i,top=is_top)
                else:
                    anchors = M.node[n]['anchors']
                    node = mrp_graph.add_node(anchors = anchors,top=is_top)
            else:
                raise NotImplementedError("No other node type supported for ucca")

            M.node[n]["mrp_node"]=node

        for e in M.edges:
            # only add forward relation
            h = e[0]
            d = e[1]
            role = M.edges[e]["role"]
            if UCCAGraph.is_inversed_edge(role):
                continue
            elif UCCAGraph.is_tok_edge(role) or role == ME_TAG:
                continue
            elif role.lower() == ":top" or role.lower() == "top":
                assert "mrp_node" in M.node[e[1]], "mrp_node {} not created, but visited and not constant".format(M.node[h])
                M.node[e[1]]["mrp_node"].is_top = True
            else:
                assert "mrp_node" in M.node[h], "mrp_node {} not created, but visited and not constant, {}, {}".format(M.node[h], M.node[d], role)
                assert "mrp_node" in M.node[d], "mrp_node {} not created, but visited and not constant, {}, {}".format(M.node[h], M.node[d], role)
                src = M.node[h]["mrp_node"].id
                tgt = M.node[d]["mrp_node"].id
                edge_label = role[1:]
                if "attributes" in M.edges[e] and M.edges[e]["attributes"]:
                    mrp_graph.add_edge(src, tgt, edge_label, attributes=M.edges[e]["attributes"], values=M.edges[e]["values"])
                else:
                    mrp_graph.add_edge(src, tgt, edge_label)
        return mrp_graph, M


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
            # TODO: try copy mechanism
            # probData is padded tensor, [src_len, batch, n_out]
            probData = unpack(prob)[0]
            if i== UCCA_CAT or i == UCCA_LE:
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
                # we don't have source to copy for UCCA
                h = (max_indices==copy_lemma_index).long()*copy_lemma_index
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

        def id_to_high(max_id, idx):
            out = []
            if idx == UCCA_CAT:
                for id in max_id:
                    if id < self.n_high:
                        out.append(self.dicts["ucca_high_dict"].getLabel(id))
                    else:
                        # this part is not differentiable, we leave it for determistic mapping
                        out.append(None)
            elif idx == UCCA_LE:
                for id in max_id:
                    if id < self.n_le_high:
                        out.append(self.dicts["ucca_high_le_dict"].getLabel(id))
                    else:
                        # this part is not differentiable, we leave it for determistic mapping
                        out.append(None)

            return out

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
            pos = self.dicts["ucca_target_pos_dict"].convertToLabels(max_id[:,UCCA_POS].tolist())
            cats = id_to_high(max_id[:,UCCA_CAT].tolist(), UCCA_CAT)
            pred_lemma = id_to_high(max_id[:,UCCA_LE].tolist(), UCCA_LE)
            # here always no cat the sense, we don't use the sense for semi,56 sense
            if UCCA_SENSE< max_id.size(1):
                sense = self.dicts["ucca_sense_dict"].convertToLabels(max_id[:,UCCA_SENSE].tolist())
            else:
                sense = None

            ucca_seq = self.rl.toUCCASeq(pos, snt,lemma,pred_lemma, cats,sense,ners,mwes)
            # TODO: show concepts, align

            # TODO: here adding gold concepts here.
            srl_concept,align,dependent_mark = self.fragment_to_node_converter.unpack_recategorized(ucca_seq,self.rl ,getsense,eval= not self.training)
            if len(srl_concept) == 0:
                # TODO: here force it not empty, but nodes may be empty sometimes
                srl_concept = [ UCCAUniversal.NULL_UCCAUniversal()]
                align = [0]
                dependent_mark = [0]
                aux_triples = []

            srl_concepts.append(srl_concept)
            aligns.append(align)
            out.append(ucca_seq)
            dependent_mark_batch.append(dependent_mark)

        return out,srl_concepts,aligns,dependent_mark_batch

    def graph_to_quadruples(self,graph):
        """
        from networkx.MultiDiGraph to quadruples
        """
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
                    if edge_data["role"] in self.dicts["ucca_rel_dict"] :
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
        For UCCA, there is normalizeMod
        """
        # TODO: graph connected graph given relation prob
        #batch of id
        #max_indices len,batch, n_feature
        #srcBatch  batch source
        #out batch AMRuniversal
        rel_dict = self.dicts["ucca_rel_dict"]
        def get_uni_var(concepts,id):
            """
            for ucca, id is also not important, just use the id as variable value
            """
            return concepts[id],UCCAVar(str(id))

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
                        r = rel_dict.getLabel(max_non_score_id[h_id,d_id].item())
                        # relations in rel_dict are forward argx, opx, sntx top
                        # and all backward relations.
                        # normalize mod should already be done in rel_dict
                        # we should normalize it here when connecting, make sure to not consider the same relation twices
                        r_inver = UCCAGraph.get_inversed_edge(r)
                        h,h_v = get_uni_var(concepts,h_id)
                        d,d_v = get_uni_var(concepts,d_id)
                        # TODO: for mwe
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
            root_symbol = UCCAUniversal.TOP_UCCAUniversal()
            graph.add_node(h_v, value=root_symbol, align=-1,gold=True,dep=1)
            graph.add_edge(h_v, root_v, key=":top", role=":top")
            graph.add_edge(root_v, h_v, key=":top-of", role=":top-of")

            for n,d in graph.nodes(True):
                le,pos,cat,sense,anchors = d["value"].le,d["value"].pos,d["value"].cat,d["value"].sense,d["value"].anchors
                # here align is the token index list
                if get_sense:
                    if sense == "" or sense == None:
                        sense = self.fragment_to_node_converter.get_senses(le, pos)

                anchors = []
                # convert token ids into character offset, with input source data
                tok_index = d["align"]
                if tok_index >= 0 and tok_index < len(source[ANCHOR_IND_SOURCE_BATCH]):
                    anchors.extend(source[ANCHOR_IND_SOURCE_BATCH][tok_index])

                d["value"] = UCCAUniversal(pos,cat,sense,le,anchors)
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
            # here we assuming, there is one root
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
