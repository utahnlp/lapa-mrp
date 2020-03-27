from utility.constants import *
from utility.score_helper import *
import utility.psd_utils.PSDGraph

class PSDNaiveScores(Scorer):
    def __init(self):
        super(PSDNaiveScores, self).__init__()
        self.show_error_names = ['Full Concept','Lemma Only','Sensed Lemma','Cat Only','POS Only','High Freq Only','Copy Only','Root','REL Triple given concept']

    @staticmethod
    def nonsense_concept(uni):
        return uni.le.lower(),uni.pos,uni.get_anchors_str()

    @staticmethod
    def nonanchors_concept(uni):
        return uni.le.lower(),uni.pos,uni.sense

    @staticmethod
    def nonlemma_concept(uni):
        return uni.pos,uni.sense, uni.get_anchors_str()

    @staticmethod
    def anchoronly_concept(uni):
        return uni.get_anchors_str()

    @staticmethod
    def remove_sense(uni):
        return (uni.pos,uni.get_anchors_str())

    #naive set overlapping for different kinds of relations
    def rel_scores_initial(self):
        nonsense_concept = PSDNaiveScores.nonsense_concept
        anchoronly_concept = PSDNaiveScores.anchoronly_concept
        root_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  and t[2]== ":top" else None
        root_score =  ScoreHelper("Root",filter=root_filter)
        root_score_given_concept =  ScoreHelper("Root given concept",filter=root_filter, second_filter=dynamics_filter)

        rel_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  else None
        rel_score =  ScoreHelper("Full Relation",filter=rel_filter)

        core_rel_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1]) and PSDGraph.is_core(t[2]) else None
        core_rel_score =  ScoreHelper("Core REL Triple",filter=core_rel_filter)

        non_sense_rel_filter = lambda t:(nonsense_concept(t[0]),nonsense_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  else None

        nonsense_rel_score =  ScoreHelper("Nonsense REL Triple",filter=non_sense_rel_filter)

        # anchor related
        anchoronly_rel_filter = lambda t:(anchoronly_concept(t[0]),anchoronly_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  else None
        anchoronly_rel_score =  ScoreHelper("AnchorOnly Rel Triple",filter=anchoronly_rel_filter)

        anchoronly_unlabel_filter = lambda t:(anchoronly_concept(t[0]),anchoronly_concept(t[1])) if legal_concept(t[0])  and legal_concept(t[1])  else None

        anchoronly_unlabel_score =  ScoreHelper("AnchorOnly Unlabel Rel Triple",filter=anchoronly_unlabel_filter)


        unlabeled_filter =lambda t:(legal_concept(t[0]),legal_concept(t[1])) if legal_concept(t[0])  and legal_concept(t[1]) else None

        unlabeled_rel_score =  ScoreHelper("Unlabeled Rel Triple",filter=unlabeled_filter)

        labeled_rel_score_given_concept =  ScoreHelper("REL Triple given concept",filter = rel_filter, second_filter=dynamics_filter)

        # all rel in psd, all functors here: https://ufal.mff.cuni.cz/pdt2.0/doc/manuals/en/t-layer/html/ch07.html
        return [nonsense_rel_score,rel_score, core_rel_score, anchoronly_rel_score, anchoronly_unlabel_score, root_score,root_score_given_concept,unlabeled_rel_score,labeled_rel_score_given_concept]

    #naive set overlapping for different kinds of concepts
    def concept_score_initial(self, dicts):
        nonsense_concept = PSDNaiveScores.nonsense_concept
        nonanchors_concept = PSDNaiveScores.nonanchors_concept
        nonlemma_concept = PSDNaiveScores.nonlemma_concept
        Non_Sense =  ScoreHelper("Non_Sense",filter=nonsense_concept)
        Non_Anchors =  ScoreHelper("Non_Anchors",filter=nonanchors_concept)
        Non_Lemma =  ScoreHelper("Non_Lemma",filter=nonlemma_concept)
        concept_score = ScoreHelper("Full Concept",filter=legal_concept)
        pos_score =  ScoreHelper("POS Only",filter=lambda uni:(uni.pos) if legal_concept(uni) else None)
        anchors_score =  ScoreHelper("Anchors Only",filter=lambda uni:(uni.get_anchors_str()) if legal_concept(uni) else None)
        lemma_score =  ScoreHelper("Lemma Only",filter=lambda uni: (uni.le.lower()) if legal_concept(uni) else None)
        sense_score =  ScoreHelper("Sense Only",filter=lambda uni: (uni.sense) if legal_concept(uni) else None)
        sensed_lemma_score =  ScoreHelper("Sensed Lemma",filter=lambda uni: (uni.le.lower(), uni.sense) if legal_concept(uni) else None)
        high_score =  ScoreHelper("High Freq Only",filter=lambda uni: uni.le.lower() if  uni.le.lower() in dicts["psd_high_dict"] and legal_concept(uni)  else None)
        copy_score =  ScoreHelper("Copy Only",filter=lambda uni: uni.le.lower() if  uni.le.lower() not in dicts["psd_high_dict"] and legal_concept(uni) else None)
        # add abstract frame score
        return  [Non_Sense,Non_Anchors,Non_Lemma,concept_score,pos_score,sense_score,anchors_score,lemma_score, sensed_lemma_score, high_score, copy_score]
