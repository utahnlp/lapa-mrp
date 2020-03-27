#!/usr/bin/env python3.6
# coding=utf-8
'''

AMRGraph builds on top of AMR from amr.py
representing AMR graph as graph,
and extract named entity (t1,..,tn, ner type, wiki) tuple. (we use model predicting for deciding ner type though)
Being able to apply recategorization to original graph,
which involves collapsing nodes for concept identification and unpacking for relation identification.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-28

make AMRGraph as a proxy for amr parsing, by offering a construct to transform a MRP graoh in AMRGraph.
@author: Jie Cao (jiessie.cao@gmail.com)
@since: 2019-06-07
'''
from utility.amr_utils.amr import *
from utility.constants import *
import networkx as nx

logger = logging.getLogger("mrp.utility.amr_utils.AMRGraph")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class AMRGraph(AMR):
    def __init__(self, anno, mrp_graph = None, normalize_inverses=True,
                 normalize_mod=True, tokens=None,aligns={},single_edge_only=True):
        '''
        create AMR from text or mrp_graph, and convert AMR to AMRGraph of standard representation
        @Jie, also support convert a MRP Graph to AMRGraph
        '''
        if anno != None or mrp_graph != None:
            super().__init__(anno, mrp_graph, tokens)
        else:
            raise NotImplementedError("anno and mrp_graph both are None")

        self.normalize_inverses = normalize_inverses
        self.normalize_mod = normalize_mod
        self.ners = []
        self.gold_concept = []
        self.gold_triple = []
        # networkx graph structure
        self.graph = nx.MultiDiGraph()
        self.wikis = []
        # based on triples, add all the edges.
        for h, r, d in [(h, r, d) for h, r, d in self.triples(normalize_inverses=self.normalize_inverses, normalize_mod=self.normalize_mod) if (r != ":instance" )]:
            if r == ':wiki':
                h, h_v = self.var_get_uni(h, True,(h, r, d ))
                d, d_v = self.var_get_uni(d)
                self.wikis.append(d)
                self.ners.append((h,d_v))
                continue
            elif r == ':top':
                d, d_v = self.var_get_uni(d)
                self.root = d
                # add_node(self, node_for_adding, **attr):
                # the extra topnode is also gold node in the amr
                self.graph.add_node(d, value=d_v, align=None,gold=True)
            else:
                h, h_v = self.var_get_uni(h, True,(h, r, d ))
                d, d_v = self.var_get_uni(d)
                self.graph.add_node(h, value=h_v, align=None,gold=True)
                self.graph.add_node(d, value=d_v, align=None,gold=True)
                # r is forward rel
                # iwhen only allow single edge, then add edges only when no edges exists between their nodes
                if single_edge_only and d in self.graph[h]:
                    logger.info("{},\n single_edge_only={}, multi_edges:{} and {}".format(self._anno, single_edge_only, str(self.graph[h][d]), (h_v, r,d_v)))
                    continue
                else:
                    self.graph.add_edge(h, d, key=r, role=r)
                    self.graph.add_edge(d, h, key=r+"-of",role=r + "-of")

    #alignment from copying mechanism
    def read_align(self, aligns):
        """
        prefix is key of aligns, which is a prefix, value is a amr node
        """
        for prefix in aligns:
            # i is a node, either varible or a concept
            i = self._index[prefix]
            if isinstance(i,Var):
                assert i  in self.graph.node,(self.graph.nodes(True),self.triples(normalize_inverses=self.normalize_inverses,
                                                                                  normalize_mod=self.normalize_mod),self._anno)
                self.graph.node[i]["align"] = aligns[prefix]
            else:
                # if it is not a var, use prefix name as its variable
                # if it in wikis,
                if Var(prefix) in self.wikis: continue
                assert Var(prefix) in self.graph.node,(prefix,aligns,self._index,self.graph.nodes(True),self._anno)
                self.graph.node[Var(prefix)]["align"] = aligns[prefix]


    def check_consistency(self,pre2c):
        """
        check pre2c is consistent with the original amr.  By checking its prefix, and its node str.
        """
        for prefix in pre2c:
            var = self._index[prefix]
            if not isinstance(var,Var): var = Var(prefix)
            if var in self.wikis: continue
            assert var  in self.graph.node,(prefix, "\n",pre2c,"\n",self.graph.node,"\n",self._anno)
            amr_c = self.graph.node[var]["value"]

            assert amr_c.gold_str() == pre2c[prefix],(prefix, var,amr_c.gold_str() ,pre2c[prefix],"\n",pre2c,"\n",self.graph.nodes(True))

    def get_gold(self):
        """
        for an amr, return all the gold concept and roles.
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
            if self.cannonical(r):
                assert "gold" in self.graph.node[h] and "gold" in self.graph.node[d]
                h = self.graph.node[h]["value"]
                d = self.graph.node[d]["value"]
                roles.append([h,d,r])

        root = self.graph.node[self.root]["value"]
        roles.append([AMRUniversal(BOS_WORD,BOS_WORD,NULL_WORD),root,':top'])
        return cons,roles

    def get_ners(self):
        """
        get all named entities,
        return [names, wikis, ner_type]
        """
        ners = []
        for v,wiki in self.ners:  #v is name variable
            name = None
            names = []
            for nearb in self.graph[v]:
                if any([":name" == edge_data["role"] for _, edge_data in self.graph[v][nearb].items()]):
                    name = nearb
                    break
            if name is None:
                print  (self.graph[v],self._anno)
                continue
            ner_type = self.graph.node[v]["value"]
            for node in self.graph[name]:
                for key, edge_data in self.graph[name][node].items():
                    if self.graph.node[node]["value"].cat == Rule_String and ":op" in edge_data["role"]:
                        names.append(( self.graph.node[node]["value"],int(edge_data["role"][-1])))  # (role, con,node)

            names = [t[0] for t in sorted(names,key = lambda t: t[1])]
            ners.append([names,wiki,ner_type])
        return ners


    def rely(self,o_node,n_node):
        """
        set rely relation, it is an attribute of original_node, value is the new node
        """
        # if o_nodei(original_node) already rely on some node, then don't set it
        if "rely" in self.graph.node[o_node]:
            return
        # set the n_node as the rely for o_node
        self.graph.node[o_node].setdefault("rely",n_node)

    def link(self,o_node,n_node,rel):
        """
        link o_node(original_node) and n_node(new_node), orginal-of is an array, storing all the recategorzied new nodes. orignal-of or has-original are about the relation ship between the original node and its recategorized node.
        The primary node will have attributes "original-of", if rely is true, then it will also have a "rely" attribute.
        """
        self.graph.node[o_node].setdefault("original-of",[]).append( n_node ) # for storing order of replacement
        if n_node:
            # make the opposite relation, has-original
            self.graph.node[n_node]["has-original"] = o_node  # for storing order of replacement
            self.graph.node[n_node]["align"] = self.graph.node[o_node]["align"]
            if rel: self.rely(o_node,n_node)

    def replace(self,node,cat_or_uni,aux=None,rel=False):
        """
        adding an aux node varaible, and link to the original sub graph
        rel : whether rely or not.
        """
        aux_le = self.graph.node[aux]['value'].le if aux else None

        if isinstance(cat_or_uni,AMRUniversal):
            universal = cat_or_uni
        else:
            le = self.graph.node[node]['value'].le
            universal = AMRUniversal(le, cat_or_uni, None, aux_le)  #aux_le is usually named entity type
        # create a new recategorized node
        # gold is not marked, so new recategorized node won't be used for relation identification
        # new node will have name as an universal cat
        var = Var(node._name+"_"+universal.cat)
        # add the new node, and link to the original node, didn't reduce the relation.
        self.graph.add_node(var, value=universal, align=None)
        self.link(node,var,rel)

        return var


    #get an amr universal node from a variable in AMR or a constant in AMR
    def var_get_uni(self, a, head=False,tri=None):
        """
        Given an input a, return its universal AMR node,
        a can be a variable node, or a prefix string.
        a : the variable, it can be a Var or a Prefix when it is constant
        return Var, AMRUniversal
        """
        if isinstance(a,Var):
            # if is a var, return a variable, and its actual node.
            return a, AMRUniversal(concept=self._v2c[a])
        else:
            if head:
                assert False, "constant as head" + "\n" + a + self._anno +"\n"+str(tri)
            return Var(a), AMRUniversal(concept=self._index[a])


    def __getitem__(self, item):
        return self.graph.node[item]

    #check whether the relation is in the cannonical direction
    def cannonical(self,r):
        return  ("-of" in r and not self.is_core(r)) or ("-of"  not in r and  self.is_core(r))

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
                    if n2 not in rel_index:                        print(self._anno)
                    # out is [rel_role, gold_dep_node_id]
                    out.append([r,rel_index[n2]])
        return [[self.graph.node[node]["value"],index], out]

    #return data for training concept identification or relation identification
    def node_value(self, keys=["value"], all=False):
        def concept_concept():
            """
            out: all the nodes after recategorizing
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
            # this node2index is note the same with

            """
            index = 0
            rel_index ={}
            # rel_out is in shape like [[n, d]]
            rel_out = []
            # If True, return entire node attribute dict as (n, ddict).
            # n is node, d is all the dict attributes, index is nodes index.
            for n, d in self.graph.nodes(True):
                if "gold" in d:
                    rel_out.append([n,d])
                    rel_index[n] = index
                    index += 1

            # rel_out is in shape like [[node, AMRUniversal]]
            # rel_index is dict(key = node, value= Int index)
            return rel_out,rel_index

        # out: all the original node without the recategorized node
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
                    # it is not a recategorization node
                    rel_out[i] = self.getRoles(n,index_dict,rel_index)
                else:
                    # gold node should not have a recategorized node
                    assert False , (self._anno, n, d["value"])
            assert (self.root  in rel_index),(self.graph.nodes[self.root],rel_index,self._anno)
            # only return the gold concepts, and it expanded nodes.
            # out : all the concept nodes include the recategorized one, include the top node, recatgorized nodes
            # out is an array, [[subnode1, subnode-attr], [subnode2, subnode]]
            # rel_out: all the original gold concepts, [[node, node_attr]] for every node, list its head and dependent,  [[head, headIndex], [[rel, depIndex]]]]
            # rel_index:  store the index of the node in the order of gold amr nodes
            return out,rel_out,rel_index[self.root]
        else:
            # return all predicted concepts
            return out
