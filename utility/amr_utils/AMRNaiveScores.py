from utility.constants import *
from utility.score_helper import *
from utility.amr_utils.amr import AMRUniversal

class AMRNaiveScores(Scorer):
    def __init(self):
        super(AMRNaiveScores, self).__init__()
        self.show_error_names = ['Full Concept','Lemma Only','Category Only','Frame Only', 'High Freq Only', 'Copy Only','Root','REL Triple given concept']

    @staticmethod
    def nonsense_concept(uni):
        return (uni.cat,uni.le) if not uni.le in Special and  not uni.cat in Special else None

    @staticmethod
    def remove_sense(uni):
        return (uni.cat,uni.le)

    #naive set overlapping for different kinds of relations
    def rel_scores_initial(self):
        nonsense_concept = AMRNaiveScores.nonsense_concept
        root_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0])  and legal_concept(t[1])  and nonsense_concept(t[0]) == (BOS_WORD,BOS_WORD) else None

        root_score =  ScoreHelper("Root",filter=root_filter)

        rel_filter = lambda t:(legal_concept(t[0]),legal_concept(t[1]),t[2]) if legal_concept(t[0]) and legal_concept(t[1])  else None
        rel_score =  ScoreHelper("Full Relation",filter=rel_filter)

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

        return [nonsense_rel_score,rel_score,root_score,unlabeled_rel_score,labeled_rel_score_given_concept,frame_score,un_frame_score,labeled_srl_score_given_concept,unlabeled_srl_score_given_concept]

    #naive set overlapping for different kinds of concepts
    def concept_score_initial(self, dicts):
        nonsense_concept = AMRNaiveScores.nonsense_concept
        Non_Sense =  ScoreHelper("Non_Sense",filter=nonsense_concept)
        concept_score = ScoreHelper("Full Concept",filter=legal_concept)
        category_score =  ScoreHelper("Category Only",filter=lambda uni:(uni.cat) if legal_concept(uni) else None)
        lemma_score =  ScoreHelper("Lemma Only",filter=lambda uni: (uni.le) if legal_concept(uni) else None)
        frame_score =  ScoreHelper("Frame Only",filter=lambda uni: (uni.le) if legal_concept(uni) and uni.cat==Rule_Frame else None)
        frame_sense_score =  ScoreHelper("Frame Sensed Only",filter=lambda uni: (uni.le,uni.sense) if legal_concept(uni) and uni.cat==Rule_Frame else None)
        frame_non_91_score =  ScoreHelper("Frame non 91 Only",filter=lambda uni: (uni.le,uni.sense) if legal_concept(uni) and uni.cat==Rule_Frame and "91" not in uni.sense else None)
        high_score =  ScoreHelper("High Freq Only",filter=lambda uni: (uni.le,uni.cat) if  uni.le in dicts["amr_high_dict"] and legal_concept(uni)  else None)
        default_score =  ScoreHelper("Copy Only",filter=lambda uni: (uni.le,uni.cat) if  uni.le not in dicts["amr_high_dict"] and legal_concept(uni) else None)
        return  [Non_Sense,concept_score,category_score,frame_score,frame_sense_score,frame_non_91_score,lemma_score,high_score,default_score]
