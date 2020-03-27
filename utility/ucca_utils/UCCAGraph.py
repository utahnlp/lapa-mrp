#!/usr/bin/env python3.6
# coding=utf-8
'''

UCCAGraph representing UCCA graph as PTBTrees with other coreference edges

make UCCAGraph as a proxy for UCCA parsing, by offering a construct to transform a MRP graoh in UCCAGraph
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-06-07
'''
from utility.constants import *
import networkx as nx
from utility.ptb_utils.trees import *
import logging
import json
import copy

logger = logging.getLogger("mrp.utility.ucca_utils.UCCAGraph")

TOP_NODE=BOS_WORD
LEAF_NODE="LEAF"
TOK_TAG=":TOK"
# Multiword entity, include multiple word express and entity
# for all terminal nodes has multiple tokes.
ME_TAG=":ME"
INTERNAL_NODE="INTERNAL"

PTB_TOKEN_ESCAPE = {u"(": u"-LRB-",
                    u")": u"-RRB-",
                    u"{": u"-LCB-",
                    u"}": u"-RCB-",
                    u"[": u"-LSB-",
                    u"]": u"-RSB-"}

class UCCAGraph(object):
    def __init__(self, anno, mrp_graph = None, tokens = None, tok_anchors = None, pos_tokens = None, normalize_inverses=True,
                 normalize_mod=True, single_edge_only=False):
        '''
        make a UCCA matrix anno or mrp_graph into our Tree proxy with networkx
        '''
        # networkx graph structure
        self.normalize_inverses = normalize_inverses
        self.normalize_mod = normalize_mod
        self.graph = nx.MultiDiGraph()
        self.single_edge_only = single_edge_only
        if anno:
            # ucca , _anno, is a matrix [['A','B','C',],['D','E','F']]
            self._anno = anno
            # parse the original UCCA matrix into a graph
            self.mrp_graph = matrix2graph(self._anno, framework = "ucca", text = None)
            if g is None:
                raise UCCAForamtError('Well-formedness error in annotation:\n' + matrix2string(self._anno())+"\n")
            self._analyze_mrp_graph(self.mrp_graph)
            self.id = self.mrp_graph.id
        elif mrp_graph:
            # load mrp_graph into UCCAGraph
            # when loading from mrp_graph, there is no more anchors for the intermididate nodes., their anchor is the sum of their childrens
            # 1. transform every node in mrp_graph.nodes into self.nodes, with its id as variable name
            self.mrp_graph = mrp_graph
            self.id = self.mrp_graph.id
            self._anno = json.dumps(self.mrp_graph.encode())
            self._analyze_mrp_graph(self.mrp_graph,tokens, tok_anchors, pos_tokens)
        else:
            raise NotImplementedError("Both anno(UCCA Matrix) and mrp_graph is NONE")

    def _analyze_mrp_graph(self, g,tokens=None, tok_anchors= None, pos_tokens= None):
        '''
        Analyze the MRP graph produced by MRP Graph, make it into a specific UCCA Graph struct
        '''
        roots=[]
        if len(g.nodes) == 0:
            self.root = None
            logger.warn("empty graph g:{}".format(g.id))
            return

        for node in g.nodes:
            v = id2Var(node.id)
            #for ucca, mrp_graph has no anchors for the node, if no achnor, then it is a non-terminal node
            # otherwise, it is a terminal nodes
            node_v = UCCAUniversal(mrp_node=node)
            # here we still use anchors, without mapping into tokens
            self.graph.add_node(v, value=node_v, anchors=node.anchors, gold=True)
            if node.is_top:
                roots.append(v)
        # cat training.mrp_ucca | grep -oP "\"tops\": \[\d+\]" | grep ","
        # now it seems only one top nodes for each UCCA, to be verified
        # now only consider the first top node as gold top
        if len(roots) > 0:
            self.root = roots[0]
            if len(roots) != 1:
                logger.error("UCCA {} should have a single top nodes".format(g.id))
        else:
            # self.root = list(self.graph.nodes)[0]
            self.root = None   # not permit none root for now

        for edge in g.edges:
            h = id2Var(edge.src)
            h_v = self.graph.nodes[h]
            d = id2Var(edge.tgt)
            d_v = self.graph.nodes[d]
            # we forece add ":" in front of the edge label for consistent with AMR
            r = ":"+edge.lab
            # in UCCA, there is no inversed edges.
            if self.single_edge_only and d in self.graph[h]:
                logger.info("{},\n single_edge_only={}, multi_edges:{} and {}".format(self._anno, self.single_edge_only, str(self.graph[h][d]), (h_v, r, d_v)))
                continue
            else:
                if edge.attributes:
                    self.graph.add_edge(h, d, key=r, role=r, attributes=edge.attributes,values=edge.values)
                    # here we also adding the inversed relation for the connectivity for DiGraph
                    self.graph.add_edge(d, h, key=r+"-of",role=r + "-of", attributes =edge.attributes, values = edge.values)
                else:
                    self.graph.add_edge(h, d, key=r, role=r)
                    # here we also adding the inversed relation for the connectivity for DiGraph
                    self.graph.add_edge(d, h, key=r+"-of",role=r + "-of")

        # add TOP on top of the root
        self.graph.add_node(TOP_NODE, value=UCCAUniversal.TOP_UCCAUniversal(),anchors=None)
        self.graph.add_edge(TOP_NODE, self.root, key='TOP', role='TOP')
        self.graph.add_edge(self.root, TOP_NODE, key='TOP-of', role='TOP-of')

        logger.info("prefix graph_nodes:{}\n  graph_edges:{}\n".format(self.graph.nodes.data(), self.graph.edges.data()))
        self.fix_anchors(tokens, tok_anchors)
        if tok_anchors != None:
            self.toPTBTree(tokens, tok_anchors, pos_tokens)


    def fix_anchors(self, tokens, tok_anchors):
        def fix_helper(p_in_role, parent):
            children = []
            discontinuous_children = []
            # sort the edges from according the alignments of dep node
            sorted_children = sorted(list(self.graph[parent].items()), key=lambda x: self.graph.node[x[0]]['align'][0]*100 + (self.graph.node[x[0]]['align'][-1] if len(self.graph.node[x[0]]['align'])>1 else 0)  if self.graph.node[x[0]]['align'] !=None and len(self.graph.node[x[0]]['align']) > 0 else total*100)
            for c,_ in sorted_children:
                for key, edge_data in self.graph[parent][c].items():
                    if edge_data["role"].endswith("-of"):
                        continue
                    else:
                        if c in used_nodes:
                            continue
                        elif "attributes" in edge_data and "remote" in edge_data["attributes"]:
                            continue
                        else:
                            used_nodes.append(c)

                        if self.graph.node[c]['align'] == None:
                            # non-terminal nodes
                            sub_children, sub_dis_children = fix_helper((parent,edge_data['role']), c)
                            #logger.error("return parent :{}, children:{}, throwup_discontinous:{}".format(c, sorted(set(sub_children)), sub_dis_children))
                            discontinuous_children.extend(sub_dis_children)
                            len_c = len(sub_children)
                            if len_c > 0:
                                if len_c == 1:
                                    pass
                                else:
                                    # handle discontinous subchildren here.
                                    con_list, max_con_index = UCCAGraph.get_max_con_list(sub_children)

                                    for j in range(len(con_list)):
                                        if j != max_con_index:
                                            for n,_ in self.graph[c].items():
                                                # logger.error("con_list: {}, current_con:{}, current nalign:{}".format(con_list, con_list[j], self.graph.node[n]['align']))
                                                if self.graph.node[n]['align'] and all([(x in con_list[j]) for x in self.graph.node[n]['align']]):
                                                    for key_n, edge_data_n in self.graph[c][n].items():
                                                        if edge_data_n['role'].endswith("-of"):
                                                            continue
                                                        else:
                                                            #logger.error("adding discontinous:c:{}, n:{}, children_align:{},n_align:{}".format(c, n, sub_children, self.graph.node[n]['align']))
                                                            discontinuous_children.append(((parent, edge_data['role']), c, (c, edge_data_n['role']), n))
                                    # adding all nodes in max_con_index list
                                    sub_children = con_list[max_con_index]

                                self.graph.node[c]['align'] = sub_children
                            else:
                                self.graph.node[c]['align'] = None

                            children.extend(sub_children)
                        else:
                            # if it is a mulitword, but discontinuous leave node, we will raise it up as a non-terminal node, and adding two terminal nodes as its children for discontinuous hanlding
                            for i in self.graph.node[c]['align']:
                                children.append(i)

            # sort and unique children
            # redo the discontinuous, if they can be merged into parent
            children = sorted(set(children))
            throwup_discontinuous = []
            if len(discontinuous_children) > 0:
                #gap = [x for x in range(children[0], children[-1]+1) if x not in children]
                con_list = []
                con_list, max_con_index = UCCAGraph.get_max_con_list(children)
                sorted_discontinuous_children = sorted(discontinuous_children, key=lambda x: min(abs(con_list[max_con_index][0] - self.graph.node[x[3]]['align'][-1]), abs(con_list[max_con_index][-1] - self.graph.node[x[3]]['align'][0])))
                for (p_role_c, c , p_role_n, n) in sorted_discontinuous_children:
                    temp_con = sorted(set(con_list[max_con_index]+ self.graph.node[n]['align']))
                    condition = (temp_con[-1] - temp_con[0] + 1== len(temp_con))
                    #or all([x in gap for x in self.graph.node[n]['align']])
                    #logger.error("discontinous parent:{}, condition:{}, c:{}, n:{}, children_align:{},mac_con:{}, n_align:{}".format(parent, condition, c, n, children, con_list[max_con_index], self.graph.node[n]['align']))
                    if condition:
                        #if filling enlonger the con, remove it from discontinuous_children, add it to the global one
                        # add this distontinuous to parent's children
                        discontinuous.append((p_role_c, c, p_role_n, n, p_in_role, parent))
                        children = sorted(set(children + self.graph.node[n]['align']))
                        con_list, max_con_index = UCCAGraph.get_max_con_list(children)
                    else:
                        # kept it in discontinuous_children, it still didn't find a parent
                        logger.error("throwup discontinous:c:{}, n:{}, children_align:{},n_align:{}".format(c, n, children, self.graph.node[n]['align']))
                        throwup_discontinuous.append((p_role_c, c, p_role_n, n))
            # if it is, merge into children
            return sorted(set(children)), throwup_discontinuous

        total = len(tokens)
        node_value = self.node_value(keys=["value","anchors"])
        # n is node, c is it value UCCAUniversal, a is anchors
        # a here are ahchors[{'from': xx, 'to': }, {'from':, 'to': }]
        for n,c,a in node_value:
            # a here are ahchors[{'from': xx, 'to': }, {'from':, 'to': }]
            # transform anchors into token ids in snt_token array
            # align is [(i, lemma[i], pos[i])]
            align = []
            # because of tokenization
            possible_align = []
            # usually only one dict in a
            if a == None:
                self.graph.node[n]['align']=None
            else:
                for d in a:
                    start = d["from"]
                    end = d["to"]
                    # tok_anchor is an array of ahchor dict [{'from': xx, 'to': }, {'from':, 'to': }]
                    # here tokens may already be combined, then token_anchor will contain more than one dict.
                    # we need to conside the following cases:
                    # 1, a has multiple anchors, each of them is corresponding to one token
                    # 2. each of them may need to be splitted into multple tokens
                    for i, anchors in enumerate(tok_anchors):
                        # for token i, here may existed more than one original token in it
                        min_start = min([anchor["from"] for anchor in anchors])
                        max_end = max([anchor["to"] for anchor in anchors])
                        if start <= min_start and end >= max_end:
                            align.append(i)
                        elif start >= min_start and end <= max_end:
                            # within some token
                            possible_align.append((i, (start-min_start, end - start)))
            if len(align) == 0:
                if len(possible_align) != 0:
                    self.graph.node[n]['align']=[possible_align[0][0]]
                    self.graph.node[n]['partial'] = possible_align[0][1]
                    logger.error("No alignments found for node {}, {}, used posssible_align:{}, partial:{}".format(n, c, possible_align, self.graph.node[n]['partial']))
                else:
                    self.graph.node[n]['align']=None
            else:
                # handle the terminal node discontinuous here
                sorted_align = sorted(align)
                if len(sorted_align) == 1:
                    self.graph.node[n]['align']=sorted(align)
                    # when a node is within in token, we use it as align, but still note down the anchors
                else:
                    # for all multiple word, raise it, and adding ME relation
                    # discontinuous
                    self.graph.node[n]['value'].ntype = INTERNAL_NODE
                    self.graph.node[n]['value'].anchors = None
                    # leave it for future align fix
                    self.graph.node[n]['align'] = None
                    con_list, max_con_index = UCCAGraph.get_max_con_list(sorted_align)
                    for j in range(len(con_list)):
                        # for each seg, adding another node
                        new_var = UCCAVar("new"+str(con_list[j]))
                        new_anchors = []
                        for k in con_list[j]:
                            new_anchors.extend(tok_anchors[k])
                        new_value = UCCAUniversal(LEAF_NODE, new_anchors)
                        self.graph.add_node(new_var, value=new_value, anchors = new_anchors, align=con_list[j])
                        self.graph.add_edge(n, new_var, key=ME_TAG, role=ME_TAG)
                        self.graph.add_edge(new_var, n, key=ME_TAG+"-of", role=ME_TAG+"-of")

        used_nodes = [TOP_NODE]
        discontinuous=[]
        top_align, throwup_discon = fix_helper((None,""),TOP_NODE)
        if len(throwup_discon) > 0:
            for (r_h, h, r_d, d) in throwup_discon:
                logger.info("there should no throwup_discon finally: {}".format((r_h, self.graph.node[h], r_d, self.graph.node[d])))

        self.fixed_graph = self.graph.copy()
        self.fixed = []
        #logger.info("fix discontinous_tokens: {}".format(discontinuous))
        # may not be the root, should the lowest common ancestor of its left token and its right token.
        for (p_r_h, h, p_r_d, d, p_r_parent, parent) in discontinuous:
            p_h_align = self.fixed_graph.node[p_r_h[0]]['align'] if p_r_h[0] else None
            p_r_align = self.fixed_graph.node[p_r_d[0]]['align'] if p_r_d[0] else None
            p_parent_align = self.fixed_graph.node[p_r_parent[0]]['align'] if p_r_parent[0] else None
            self.fixed_graph.remove_edge(h, d)
            self.fixed_graph.remove_edge(d, h)
            self.fixed_graph.add_edge(parent, d, key=p_r_d[1], role=p_r_d[1])
            self.fixed_graph.add_edge(d, parent, key=p_r_d[1]+"-of", role=p_r_d[1]+"-of")
            self.fixed.append((((p_h_align, p_r_h[1]),self.fixed_graph.node[h]['align']),((p_r_align, p_r_d[1]), self.fixed_graph.node[d]['align']), ((p_parent_align, p_r_parent[1]), self.fixed_graph.node[parent]['align'])))
            #logger.error("Do edge changes for discontinuous:{}".format((p_r_h, self.fixed_graph.node[h]['align'], p_r_d, self.fixed_graph.node[d]['align'], p_r_parent, self.fixed_graph.node[parent]['align'])))

        #logger.error("after fix graph: for {}\n fixed_grpah_nodes:{}\n  fixed_graph_edges:{}\n".format(self.id, self.fixed_graph.nodes.data(), self.fixed_graph.edges.data()))


    @staticmethod
    def get_max_con_list(sorted_list):
        len_c = len(sorted_list)
        max_con_size = 0
        max_con_index = 0
        continuous = []
        con_list = []
        for i in range(len_c):
            continuous.append(sorted_list[i])
            if sorted_list[i] + 1 not in sorted_list:
                con_list.append(copy.deepcopy(continuous))
                # use right most continuous
                if len(continuous) >= max_con_size:
                    max_con_index = len(con_list) - 1
                    max_con_size = len(continuous)
                continuous = []
        return con_list, max_con_index


    def toPTBTree(self, tokens, tok_anchors, pos_tokens):
        def helper(p_in_role, parent):
            children = []
            sorted_children = sorted(list(self.fixed_graph[parent].items()), key=lambda x: self.fixed_graph.node[x[0]]['align'][0]*100 + (self.fixed_graph.node[x[0]]['align'][-1] if len(self.fixed_graph.node[x[0]]['align'])>1 else 0) + (self.fixed_graph.node[x[0]]['partial'][0] if 'partial' in self.fixed_graph.node[x[0]] else 0) if self.fixed_graph.node[x[0]]['align'] !=None and len(self.fixed_graph.node[x[0]]['align']) > 0 else total*100)
            for c,_ in sorted_children:
                for key, edge_data in self.fixed_graph[parent][c].items():
                    c_in_role = (self.fixed_graph.node[parent]['align'], edge_data["role"])
                    if edge_data["role"].endswith("-of"):
                        continue
                    else:
                        if c in used_nodes:
                            # it retrnace, then can be remote
                            # skip the remote and original attribute
                            reents.append(((p_in_role, self.fixed_graph.node[parent]['align']), (used_nodes[c], self.fixed_graph.node[c]['align']), edge_data))
                            continue
                        elif "attributes" in edge_data and "remote" in edge_data["attributes"]:
                            # not in used, but remote nodes, delay the processing
                            unresolved_remotes.setdefault(c,[]).append(((p_in_role, self.fixed_graph.node[parent]['align']), edge_data))
                            continue

                        else:
                            used_nodes[c] = c_in_role
                            if c in unresolved_remotes:
                                for (p_in_role_align, pre_edge) in unresolved_remotes[c]:
                                    reents.append((p_in_role_align, (used_nodes[c], self.fixed_graph.node[c]['align']), pre_edge))
                                unresolved_remotes[c] = []

                        if self.fixed_graph.node[c]['value'].ntype == INTERNAL_NODE:
                            # non-terminal nodes
                            sub_children = helper(c_in_role, c)
                            if len(sub_children)==0:
                                logger.error("empty children: {}, {}".format(c, self.fixed_graph[c]))
                                # children has been visited
                                # now let's just drop the node, it is not supported by ucca
                                pass
                            else:
                                # handle uncontinous issue
                                children.append(InternalTreebankNode(edge_data["role"], sub_children))
                        else:
                            # terminal nodes
                            # get the underlay tokens,
                            # it is already sorted
                            sub_children = []

                            if self.fixed_graph.node[c]['align']:
                                # for every word in the continuous seg, make
                                if 'partial' in self.fixed_graph.node[c]:
                                    whole_word = ""
                                    for i in self.fixed_graph.node[c]['align']:
                                        start = self.fixed_graph.node[c]['partial'][0]
                                        l = self.fixed_graph.node[c]['partial'][1]
                                        escaped_word = PTBTOKEN_ESCAPE[tokens[i][start:start+l]] if tokens[i][start:start+l] in PTB_TOKEN_ESCAPE else tokens[i][start:start+l]
                                        leaf_tok = LeafTreebankNode(TOK_TAG, escaped_word)
                                        sub_children.append(leaf_tok)
                                        leave_tok_nodes.append(leaf_tok)
                                        whole_word+=tokens[i][start:start+l]
                                    leave_whole_token.append(whole_word)
                                else:
                                    for i in self.fixed_graph.node[c]['align']:
                                        escaped_word = PTBTOKEN_ESCAPE[tokens[i]] if tokens[i] in PTB_TOKEN_ESCAPE else tokens[i]
                                        leaf_tok = LeafTreebankNode(TOK_TAG, escaped_word)
                                        sub_children.append(leaf_tok)
                                        leave_tok_nodes.append(leaf_tok)
                                        leave_whole_token.append(tokens[i])
                                children.append(InternalTreebankNode(edge_data["role"], sub_children))

            branch_total.append(len(children))
            # already sorted
            return children

        # sort the nodes by anchors
        # build the tree from the top
        total = len(tokens)
        leave_tok_nodes = []
        leave_whole_token = []
        reents= []
        unresolved_remotes = {}
        branch_total = [1]
        # n: role

        used_nodes = {TOP_NODE:(None,"")}
        trees = helper((None,""), TOP_NODE)
        assert len(trees) > 0,"empty tree for {}".format(self.id)
        self.tree = trees[0]
        ptb_snt = " ".join([x.word for x in leave_tok_nodes])
        ori_snt = " ".join([x for x in tokens])
        branch_total = sum(branch_total)
        if branch_total != len(self.fixed_graph.nodes):
            logger.error("reentrance = {}, branch_total={}, graph_nodes={}, in {}".format(reents, branch_total, len(self.fixed_graph.nodes), self.id))
        if len(reents) + len(self.fixed_graph.nodes) - 1 != len(self.fixed_graph.edges)/2:
            logger.error("tree error: {}+{} - 1 !={}".format(len(reents), len(self.fixed_graph.nodes), len(self.fixed_graph.edges)/2))
        self.reents = reents
        if ptb_snt != ori_snt:
            logger.error("ptb tree is not equal to the orignal sentence,{},\n {} \n {}".format(self.tree.linearize(), ptb_snt, ori_snt))
            self.tree = None
            return None
        else:
            return self.tree

    def get_gold(self):
        """
        for a UCCAGraph, return all the gold concept and roles.
        """
        cons = []
        roles = []
        for n, d in self.graph.nodes(True):
            # add gold concepts into a list
            if "gold" in d:
                v = d["value"]
                cons.append(v)

        for h, d, _, rel in self.graph.edges(keys=True,data=True):
            # add roles into a list, every role is [h, d, r]
            r = rel["role"]
            # during learning, only predict the cannonical edge labels, its inversed version is only for connectivity of DiGraph
            if self.cannonical(r):
                assert "gold" in self.graph.node[h] and "gold" in self.graph.node[d]
                h = self.graph.node[h]["value"]
                d = self.graph.node[d]["value"]
                roles.append([h,d,r])

        if self.root:
            root = self.graph.node[self.root]["value"]
            # todo: add a special Node for UCCA
            roles.append([UCCAUniversal.TOP_UCCAUniversal(),root,':TOP'])

        # WARN: here roles may not contains any top relations
        return cons,roles

    def __getitem__(self, item):
        return self.graph.node[item]

    #check whether the relation is in the cannonical direction
    # ARG0, ARG1, ... ARGn as core rel
    # BV also as core rel.
    # compound and mwe are special relation, which are usually happened in consecutive tokens.
    # Now also make the model to learn this.
    def cannonical(self,r):
        # now all rel are core, and all forward are cannonical
        return ("-of"  not in r and  self.is_core(r))

    @staticmethod
    def is_core(r):
        """
        for UCCA, now treat all edge as core rel
        """
        return ("-of" not in r)

    @staticmethod
    def is_tok_edge(r):
        return r.startswith(TOK_TAG)

    @staticmethod
    def is_unk_edge(r):
        return "UNK" in r

    @staticmethod
    def is_inversed_edge(role):
        return role.endswith("-of")

    @staticmethod
    def is_inversed_edge(edge):
        if edge.endswith("-of"):
            return True
        else:
            return False

    @staticmethod
    def get_inversed_edge(edge):
        if edge.endswith("-of"):
            inverse = edge[:-3]
        else:
            inverse = edge + "-of"
        return inverse

    @staticmethod
    def get_normalizede_edge(edge):
        if is_inversed_edge(edge):
            return get_inversed_edge(edge)
        else:
            return edge

    def getRoles(self,node,index_dict,rel_index,relyed = None):
        """
        get all the roles of a node
        node : node variable
        index_dict is dict(key=node, value = intIndex), the index is the index for recategorized nodes
        rel_index, is dict(key=node, value = intIndex), the index is the index for gold nodes
        return [[node, node1ReCateIndex], [[rel, node2GoldIndex]]]]
        """
        # (amruniversal,index,[[role,rel_index]])
        if relyed and relyed not in index_dict:
            print ("rely",node,relyed,self.graph.node[relyed]["value"],index_dict,self._anno)
        elif relyed is None and node not in index_dict: print (self.graph.node[node]["value"])
        # get only the original node index, this index is recategorised index
        index = index_dict[node] if relyed is None else index_dict[relyed]
        out = []
        #   if self.graph.node[node]["value"].le != "name":
        # self.graph[node] returns all the adj node in a dict(key=neighbor, value=attributes)
        for n2 in  self.graph[node]:
            # the role from n2 to node
            for key, edge_data in self.graph[node][n2].items():
                r = edge_data["role"]
                if self.cannonical(r):
                    if n2 not in rel_index:
                        print(self._anno)
                    # out is [rel_role, gold_dep_node_id]
                    out.append([r,rel_index[n2]])
        return [[self.graph.node[node]["value"],index], out]

    #return data for training concept identification or relation identification
    def node_value(self, keys=["value"], all=False):
        def concept_concept():
            """
            out: all nodes after recategorizing
            index_dict, [key:node, value: index], index is the order for transduce the node in the AMR graph.
            """
            # out is an array, [[subnode1, subnode-attr], [subnode2, subnode]]
            out = []
            # index the order id of a node.
            index = 0
            # save a node2index dict
            index_dict ={}
            for n, d in self.graph.nodes(True):
                # https://networkx.github.io/documentation/networkx-2.1/reference/classes/generated/networkx.Graph.nodes.html
                # n is the node, d is the data with all attributes
                # graph.nodes(True), means return entire node attribute dict
                # if it has recategorized new nodes, iterate its recategorizeed nodes, only add the combination nodes, not the original node
                if "original-of" in d:
                    comps = d["original-of"]
                    for comp in comps:
                        if comp is None:
                            continue
                        comp_d = self.graph.node[comp]
                        # output a (node, value1, value2)
                        # by default key is value, which is AMRUniversal Node of that node.
                        out.append([comp] + [comp_d[k] for k in keys])
                        index_dict[comp] = index
                        index += 1
                elif not ("has-original" in d or  "rely" in d):
                    # TODO: all node in UCCA is the original node, without categorizing
                    # not a recategorized node, just use that node itself.
                    out.append([n] + [d[k] for k in keys])
                    index_dict[n] = index
                    index += 1
            # out is an array, [[subnode1, subnode-attr], [subnode2, subnode2_attr]]
            # index_dict is dict(key=node, value = intIndex)
            return out,index_dict

        def rel_concept():
            """
            return the gold node and its node2index dict
            """
            index = 0
            rel_index ={}
            # rel_out is in shape like [[n, d]]
            rel_out = []
            # If True, return entire node attribute dict as (n, ddict).
            # n is node Varible, d is all the dict attributes, index is nodes index.
            for n, d in self.graph.nodes(True):
                if "gold" in d:
                    rel_out.append([n,d])
                    rel_index[n] = index
                    index += 1

            # rel_out is in shape like [[var, Node]]
            # rel_index is dict(key = var, value= Int index)
            return rel_out,rel_index

        # out: all the nodes after recategorization
        # index_dict, [key:node, value: index], index is the order for transduce the node in the AMR graph.
        out,index_dict = concept_concept()
        if all:
            # all means all attributes
            # rel_out: all the gold concepts
            # rel_index: a different index from gold node transduce order.
            rel_out, rel_index = rel_concept()
            for i, n_d in enumerate( rel_out):
                n,d = n_d
                # rely means n is a original node
                if "rely" in d:
                    # [[node, node1ReCateIndex], [[rel, node2GoldIndex]]]]
                    rel_out[i] = self.getRoles(n,index_dict,rel_index,d["rely"])
                elif not ("has-original" in d or  "original-of" in d):
                    # it is an original node
                    # UCCA will follow this path
                    rel_out[i] = self.getRoles(n,index_dict,rel_index)
                else:
                    # gold node should not have a recategorized node
                    assert False , (self._anno, n, d["value"])

            if self.root:
                # if there is root, then root index must be in the roots.
                assert (self.root in rel_index),(self.graph.nodes[self.root],rel_index,self._anno)
                root_index = rel_index[self.root]
            else:
                root_index = None
            # only return the gold concepts, and it expanded nodes.
            # out : all the concept nodes include the recategorized one, include the top node, recatgorized nodes
            # out is an array, [[subnode1, subnode-attr], [subnode2, subnode]]
            # rel_out: all the original gold concepts, [[node, node_attr]] for every node, list its head and dependent,  [[head, headIndex], [[rel, depIndex]]]]
            # rel_index:  store the index of the node in the order of gold amr nodes
            return out,rel_out,root_index
        else:
            # return all predicted concepts
            return out

class UCCAFormatError(Exception):
    pass

def matrix2string(matrix):
    return '\n'.join(['\t'.join(row) for row in matrix])

def id2Var(id):
    return UCCAVar("id"+str(id))

class UCCAVar(object):
    """
    In this AMR file, nodes are classified into 3 classes: Concept, Var, Constant
    The variable class, used for variable for a name, representing the reentranies.
    """
    def __init__(self, name):
        self._name = name

    # simply override the less equal than
    def __le__(self,other):
        return self._name < other._name

    def is_var(self):
        return True

    def is_concept(self):
        return False

    def is_constant(self):
        return False

    def __repr__(self):
        return 'UCCAVar(' + self._name +')'

    # override the string into its name, name is the identity of the variable node.
    def __str__(self):
        return self._name

    def __call__(self, **kwargs):
        return self.__str__()

    def __eq__(self, that):
        return isinstance(that, type(self)) and self._name == that._name

    def __hash__(self):
        return hash(self._name)

    def deepcopy(self,s=""):
        return UCCAVar(self._name+s)

#decompose ucca node to le_pos_cat_sense
def decompose(c):
    """
    extract le, pos and cat from a single ucca node
    """
    if c is None: return None
    if isinstance(c,UCCAUniversal):
        return c.ntype, c.anchors
    return None, None

class UCCAUniversal(object):
    def __init__(self, *args, **kwargs):
        # *args for un-named argeument, kwarges are keyworded arguments
        if "string" in kwargs:
            raise NotImplementedError("construact from string is not supported on UCCA")
        elif "mrp_node" in kwargs:
            self.construct_by_mrp_node(kwargs["mrp_node"])
        elif "ptb_node" in kwargs:
            self.construct_by_ptb_node(kwargs["ptb_node"])
        else:
            # ntype, anchors
            self.construct_by_content(args[0], args[1])

    @staticmethod
    def TOP_UCCAUniversal():
        return UCCAUniversal(TOP_NODE, None)

    @staticmethod
    def NULL_UCCAUniversal():
        return UCCAUniversal(None, None)

    def construct_by_mrp_node(self, node):
        if node.anchors:
            self.anchors = node.anchors
            self.ntype = LEAF_NODE
        else:
            self.anchors = None
            self.ntype = INTERNAL_NODE

    def construct_by_content(self, ntype, anchors=None):
        """
        NULL_WORD is empty string, which is just place holder but not break the structure a node
        """
        self.ntype = ntype
        self.anchors = anchors

    def get_anchors_str(self):
        if self.anchors:
            return ",".join(["{}-{}".format(x["from"],x["to"]) for x in self.anchors])
        else:
            return " - "

    def __repr__(self):
        return "{}({})".format(self.ntype, self.get_anchors_str())

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return isinstance(other, UCCAUniversal) and self.ntype == other.ntype and self.anchors == other.anchors
