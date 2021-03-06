from utility.constants import *
from utility.score_helper import *
import utility.eds_utils.EDSGraph

class EDSNaiveScores:
    @staticmethod
    def nonsense_concept(uni):
        return uni.le,uni.pos,uni.predicate,uni.get_anchors_str()

    @staticmethod
    def nonanchors_concept(uni):
        return uni.le,uni.pos,uni.predicate,uni.sense

    @staticmethod
    def nonlemma_concept(uni):
        return uni.pos,uni.predicate,uni.sense, uni.get_anchors_str()

    @staticmethod
    def nonpredicate_concept(uni):
        return uni.le, uni.pos,uni.sense, uni.get_anchors_str()

    @staticmethod
    def remove_sense(uni):
        return (uni.pos,uni.predicate,uni.get_anchors_str())

    #naive set overlapping for different kinds of relations
    def rel_scores_initial(self):
        nonsense_concept = EDSNaiveScores.nonsense_concept
        root_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  and t[2]== ":top" else None
        root_score =  ScoreHelper("Root",filter=root_filter)
        root_score_given_concept =  ScoreHelper("Root given concept",filter=root_filter, second_filter=dynamics_filter)

        rel_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  else None
        rel_score =  ScoreHelper("REL Triple",filter=rel_filter)

        non_sense_rel_filter = lambda t:(nonsense_concept(t[0]),nonsense_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  else None
        nonsense_rel_score =  ScoreHelper("Nonsense REL Triple",filter=non_sense_rel_filter)

        unlabeled_filter =lambda t:(legal_concept(t[0]),legal_concept(t[1])) if legal_concept(t[0])  and legal_concept(t[1]) else None

        unlabeled_rel_score =  ScoreHelper("Unlabeled Rel Triple",filter=unlabeled_filter)

        labeled_rel_score_given_concept =  ScoreHelper("REL Triple given concept",filter = rel_filter, second_filter=dynamics_filter)


        un_srl_filter =lambda t:(legal_concept(t[0]),legal_concept(t[1])) if legal_concept(t[0])  and legal_concept(t[1]) and t[2].startswith(':ARG') else None

        un_frame_score =  ScoreHelper("Unlabled SRL Triple",filter=un_srl_filter)

        srl_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1]) and t[2].startswith(':ARG') else None
        frame_score =  ScoreHelper("SRL Triple",filter=srl_filter)

        labeled_srl_score_given_concept =  ScoreHelper("SRL Triple given concept",filter = srl_filter, second_filter=dynamics_filter)

        unlabeled_srl_score_given_concept =  ScoreHelper("Unlabeled SRL Triple given concept",filter = un_srl_filter, second_filter=dynamics_filter)

        return [nonsense_rel_score,rel_score,root_score,root_score_given_concept,unlabeled_rel_score,labeled_rel_score_given_concept,frame_score,un_frame_score,labeled_srl_score_given_concept,unlabeled_srl_score_given_concept]

    #naive set overlapping for different kinds of concepts
    def concept_score_initial(self, dicts):
        nonsense_concept = EDSNaiveScores.nonsense_concept
        nonanchors_concept = EDSNaiveScores.nonanchors_concept
        nonlemma_concept = EDSNaiveScores.nonlemma_concept
        nonpredicate_concept = EDSNaiveScores.nonpredicate_concept
        Non_Sense =  ScoreHelper("Non_Sense",filter=nonsense_concept)
        Non_Anchors =  ScoreHelper("Non_Anchors",filter=nonanchors_concept)
        Non_Lemma =  ScoreHelper("Non_Lemma",filter=nonlemma_concept)
        Non_Predicate =  ScoreHelper("Non_Predicate",filter=nonpredicate_concept)
        concept_score = ScoreHelper("Full Concept",filter=legal_concept)
        pos_score =  ScoreHelper("POS Only",filter=lambda uni:(uni.pos) if legal_concept(uni) else None)
        anchors_score =  ScoreHelper("Anchors Only",filter=lambda uni:(uni.get_anchors_str()) if legal_concept(uni) else None)
        lemma_score =  ScoreHelper("Lemma Only",filter=lambda uni: (uni.le) if legal_concept(uni) else None)
        frame_score =  ScoreHelper("Frame Only",filter=lambda uni: (uni.predicate) if legal_concept(uni) else None)
        sense_score =  ScoreHelper("Sense Only",filter=lambda uni: (uni.sense) if legal_concept(uni) else None)
        frame_sense_score =  ScoreHelper("Frame Sensed Only",filter=lambda uni: (uni.predicate,uni.sense) if legal_concept(uni) else None)
        # add abstract frame score
        return  [Non_Sense,Non_Anchors,Non_Lemma,Non_Predicate,concept_score,pos_score,frame_score,sense_score,frame_sense_score,anchors_score,lemma_score]
