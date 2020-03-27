from utility.constants import *
from utility.amr_utils.amr import *
from utility.dm_utils.DMGraph import *
from utility.psd_utils.PSDGraph import *
import logging

score_logger = logging.getLogger("mrp.score")

def list_to_mulset(l):
    s = dict()
    for i in l:
        if isinstance(i,AMRUniversal) and i.le == "i"and i.cat == Rule_Concept :
            s[i] = 1
        else:
            s[i] = s.setdefault(i,0)+1
    return s

def legal_concept(uni):
    if isinstance(uni,AMRUniversal):
        return (uni.cat,uni.le.lower(),uni.sense) if not uni.le in Special and  not uni.cat in Special else None
    elif isinstance(uni, DMUniversal):
        return (uni.pos,uni.le.lower(), uni.cat, uni.sense, uni.get_anchors_str()) if not uni.le in Special and not uni.pos in Special else None
    elif isinstance(uni, PSDUniversal):
        return (uni.pos,uni.le.lower(),uni.sense, uni.get_anchors_str()) if not uni.le in Special and not uni.pos in Special else None
    else:
        return uni

def dynamics_filter(triple,concept_seq):
    """
       triple, check node sin concept_seq, BOS_WORD is TOP
    """
    if triple[0] in concept_seq and triple[1] in concept_seq or BOS_WORD in triple[0]:
        return triple[:3]
    return None


def P_R_F1(T,P,TP):
    if TP == 0:
        return 0,0,0
    P = TP/P
    R = TP/T
    F1 = 2.0/(1.0/P+1.0/R)
    return P,R,F1

class Scorer:
    def __init__(self):
        self.concept_only_scorer_names = ["Full Concept"]
        self.smatch_scorer_names = ["Full Concept","Full Relation"]

    def get_smatch(self, concept_scores, rel_scores, rel=True):
        concept_only_scorers = []
        smatch_scorers = []
        for s in concept_scores + rel_scores:
            if s.name in self.concept_only_scorer_names:
                concept_only_scorers.append(s)

            if s.name in self.smatch_scorer_names:
                smatch_scorers.append(s)
        return self.get_score(smatch_scorers) if rel else self.get_score(concept_only_scorers)

    def get_score(self, scorers):
        return P_R_F1(*[sum(m) for m in zip(*[(x.t_p_tp[0], x.t_p_tp[1], x.t_p_tp[2]) for x in scorers])])

class ScoreHelper:

    def __init__(self,name, filter ,second_filter=None, concept_list_to_multiset=list_to_mulset):
        self.t_p_tp = [0,0,0]
        self.name = name
        self.f = filter
        self.second_filter = second_filter
        self.concept_list_to_multiset = concept_list_to_multiset
        self.false_positive = {}
        self.false_negative = {}

    def T_P_TP_Batch(self,hypos,golds,accumulate=True,second_filter_material =None):
        if self.second_filter:
            T,P,TP,fp,fn = ScoreHelper.static_T_P_TP_Batch(hypos,golds,self.concept_list_to_multiset, self.f,self.second_filter,second_filter_material)
        else:
    #        assert self.name != "Unlabled SRL Triple",(hypos[-20],"STOP!",golds[-20])
            T,P,TP,fp,fn = ScoreHelper.static_T_P_TP_Batch(hypos,golds,self.concept_list_to_multiset, self.f)
        if accumulate:
            self.add_t_p_tp(T,P,TP)
            self.add_content(fp,fn)
        return T,P,TP

    def add_t_p_tp(self,T,P,TP):
        self.t_p_tp[0] += T
        self.t_p_tp[1] += P
        self.t_p_tp[2] += TP

    def add_content(self,fp,fn ):
        for i in fp:
            self.false_positive[i] = self.false_positive.setdefault(i,0)+1
        for i in fn:
            self.false_negative[i] = self.false_negative.setdefault(i,0)+1

    def show_error(self,t = 5):
        score_logger.info("{}, false_positive: {}".format(self.name, [(k,self.false_positive[k]) for  k in sorted(self.false_positive,reverse=True,key=self.false_positive.get) if self.false_positive[k]> t]))
        score_logger.info("{}, false_negative: {}".format(self.name, [(k,self.false_negative[k]) for  k in sorted(self.false_negative,reverse=True,key=self.false_negative.get) if self.false_negative[k]>t]))

    def get_error(self,t = 5):
        false_positive = "{}, false_positive: {}".format(self.name, [(k,self.false_positive[k]) for  k in sorted(self.false_positive,reverse=True,key=self.false_positive.get) if self.false_positive[k]> t])
        false_negative = "{}, false_negative: {}".format(self.name, [(k,self.false_negative[k]) for  k in sorted(self.false_negative,reverse=True,key=self.false_negative.get) if self.false_negative[k]>t])
        return false_positive, false_negative

    def __str__(self):
        s = self.name+", [T,P,TP: "+ " ".join([str(i) for i in  self.t_p_tp])+"], [P,R,F1: "+ " ".join(["{0:.4f}".format(i) for i in  P_R_F1(*self.t_p_tp)]) + "]"
        return s

    @staticmethod
    def filter_seq(filter,seq):
        out = []
        for t in seq:
            filtered = filter(t)
            if filtered  and  filtered[0] != BOS_WORD and filtered != BOS_WORD:
                out.append(filtered)
        return out


    @staticmethod
    def T_TP_Seq(hypo,gold, list_to_multiset, filter,second_filter = None,second_filter_material = None):
        filter_seq = ScoreHelper.filter_seq
        gold = filter_seq(filter,gold)
        hypo = filter_seq(filter,hypo)
        fp = []
        fn = []
        if second_filter:  #only for triple given concept
            second_filter_predicated = filter_seq(legal_concept, second_filter_material[0])
            second_filter_with_material = lambda x: second_filter(x,second_filter_predicated)
            gold = filter_seq(second_filter_with_material,gold)

            second_filter_gold = filter_seq(legal_concept, second_filter_material[1])
            second_filter_with_material = lambda x: second_filter(x,second_filter_gold)

            hypo = filter_seq(second_filter_with_material,hypo)

        TP = 0
        T = len(gold)
        P = len(hypo)
        gold = list_to_mulset(gold)
        hypo = list_to_mulset(hypo)

        for d_g in gold:
            if d_g in hypo :
                TP += min(gold[d_g],hypo[d_g])
                fn = fn + [d_g] *min(gold[d_g]-hypo[d_g],0)
            else:
                fn = fn + [d_g] *gold[d_g]

        for d_g in hypo:
            if d_g in gold :
                fp = fp + [d_g] *min(hypo[d_g]-gold[d_g],0)
            else:
                fp = fp + [d_g] *hypo[d_g]
        return T,P,TP,fp,fn


    @staticmethod
    def static_T_P_TP_Batch(hypos,golds,list_to_multiset, filter,second_filter=None,second_filter_material_batch = None):
        TP,T,P = 0,0,0
        FP,FN = [],[]
        assert hypos, golds
        for i in range(len(hypos)):
            if second_filter:
                t,p,tp,fp,fn = ScoreHelper.T_TP_Seq(hypos[i],golds[i],list_to_multiset, filter,second_filter,(second_filter_material_batch[0][i],second_filter_material_batch[1][i]))
            else:
                t,p,tp,fp,fn = ScoreHelper.T_TP_Seq(hypos[i],golds[i],list_to_multiset, filter)
            T += t
            P +=p
            TP += tp
            FP += fp
            FN += fn
        return T,P,TP,FP,FN

    @staticmethod
    def filter_mutual(hypo,gold,mutual_filter):
        filter_seq=ScoreHelper.filter_seq
        mutual_filter=ScoreHelper.mutual_filter
        filtered_hypo = [item for sublist in filter_seq(mutual_filter,hypo) for item in sublist]
        out_hypo = []
        filtered_gold = [item for sublist in filter_seq(mutual_filter,gold) for item in sublist]
        out_gold = []

        for data in hypo:
            d1,d2 = mutual_filter(data)
            if d1 in filtered_gold and d2 in filtered_gold:
                out_hypo.append(data)

        for data in gold:
            d1,d2 = mutual_filter(data)
            if d1 in filtered_hypo and d2 in filtered_hypo:
                out_gold.append(data)

        return out_hypo,out_gold
