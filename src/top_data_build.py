#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts build dictionary and data into numbers, and seralize into pickle file.

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-07-30
'''

from utility.data_helper import *
from utility.top_utils import *
from utility.ptb_utils import *
from utility.constants import *
from parser.Dict import *
from parser.DMProcessors import *
import logging
from parser.modules.bert_utils import MRPBertTokenizer
from parser.modules.char_utils import CharTokenizerUtils

import argparse

logger = logging.getLogger("mrp")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def top_data_build_parser():
    parser = argparse.ArgumentParser(description='top_data_build.py')

    ## Data options
    parser.add_argument('--threshold', default=0, type=int,
                        help="""threshold for high frequency concepts""")

    parser.add_argument('--skip', default=0, type=int,
                        help="""skip dict build if dictionary already built""")
    parser.add_argument('--merge_common_dicts', default=1, type=int,
                        help="""whether to merge common dict if already existed""")
    parser.add_argument('--suffix', default=".mrp_top", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--companion_suffix', default=".mrp_conllu_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--build_folder', default="", type=str,
                        help="""the build folder for dict and rules, data""")
    # https://raw.githubusercontent.com/huggingface/pytorch-transformers/master/pytorch_transformers/tokenization_bert.py
    parser.add_argument('--bert_model', default="bert-base-cased", type=str,
                        help="""the name for pretraind bert name""")
    parser.add_argument('--do_lower_case', default=False, type=bool,
                        help="""Whether to lower case when tokenization""")
    return parser


parser = top_data_build_parser()
opt = parser.parse_args()
# here use bert to preprare token ids, used for fine tuning.
# if for static bert, we can do that offline, save an encoding for each utterance
bert_tokenizer = MRPBertTokenizer.from_pretrained(opt.bert_model, do_lower_case=opt.do_lower_case)

suffix = opt.suffix
trainFolderPath = opt.build_folder + "/training/"
trainingPickleFile = opt.build_folder + "/training/training.top_pickle_processed"
trainingFilesPath = folder_to_files_path(trainFolderPath, suffix)
trainingCompanionFilesPath = folder_to_files_path(trainFolderPath, opt.companion_suffix)

devFolderPath = opt.build_folder + "/dev/"
devPickleFile = opt.build_folder + "/dev/dev.top_pickle_processed"
devFilesPath = folder_to_files_path(devFolderPath, suffix)
devCompanionFilesPath = folder_to_files_path(devFolderPath, opt.companion_suffix)

testFolderPath = opt.build_folder + "/test/"
testPickleFile = opt.build_folder + "/test/test.top_pickle_processed"
testFilesPath = folder_to_files_path(testFolderPath, suffix)
testCompanionFilesPath = folder_to_files_path(testFolderPath, opt.companion_suffix)

def mytop_to_seq(top, snt_token, lemma_token, pos, tok_anchors, ners, mwes, rl, fragment_to_node_converter):  # high_freq should be a dict()
    """
    make a top to return a squence of node and rels
    # output_concepts : [DM_CAT, DM_LE, DM_NER(DM_AUX), DM_SENSE, DM_CAN_COPY, aligments]
    # rel: [rel_feature, rel_tgt, root_id], unaligned_inx
    # rel_feature:[[DM_CAT, DM_LE, DM_NER, DM_SENSE, node1Index], the node1index is the index acoording to the order of tanversing the recategorized nodes.
    # rel_tgt: [[rel,node2Index], [rel2, node2index]], the index the gold index
    # root_id is the index in gold_top_index
    # unaligned_index = [index], this index is the index in the snt
    # return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]
    """

    def uni_to_list(uni, can_copy=0, can_le_copy=0):
        le = uni.le
        pos = uni.pos
        cat = uni.cat
        sense = uni.sense
        data = [0, 0, 0, 0, 0, 0]
        data[DM_POS] = pos
        data[DM_CAT] = cat
        data[DM_SENSE] = sense
        data[DM_LE] = le
        data[DM_CAN_COPY] = 1 if can_copy else 0
        data[DM_LE_CAN_COPY] = 1 if can_le_copy else 0
        return data

    output_concepts = []
    lemma_str = " ".join(lemma_token)
    # do align before converting., add align
    rl.get_matched_concepts(snt_token,top,lemma_token, pos, tok_anchors, ners, mwes)
    # do recategorization
    fragment_to_node_converter.convert(top, rl, snt_token, lemma_token, lemma_str)
    # return all concepts, relations, rootid
    # concepts = [[subnode1, subnode-attr], [subnode2, subnode-attr]]
    # rel = [[node, node1ReCateIndex], [[rel, node2GoldIndex]]]], the order is accroding to the index of gold top index
    # root_id, int, the orderid in all original gold nodes
    concepts, rel, root_id = top.node_value(keys=["value", "align"], all=True)

    results=rl.get_matched_concepts(snt_token,top,lemma_token, pos, tok_anchors, ners, mwes)
    # use rules to match and get all alignments for copy
    # return [[n,c,a], ...], n is variable, c is DMUniversal, a is alignment[(i,lemma[i],pos[i])]
    aligned_index = []
    n_top = len(concepts)
    n_snt = len(snt_token)
    l = len(lemma_token) if lemma_token[-1] != "." else len(lemma_token) - 1

    for i, n_c_a_f in enumerate(results):
        uni = n_c_a_f[1]
        align = n_c_a_f[2]
        aligned_index += align

        # CAN_COPY is decided by the possible alignment from the string rules, which can find the possible alignments.
        data = uni_to_list(uni, n_c_a_f[3], n_c_a_f[4])
        # last dimension is alignements, an array of indices can align to the node.
        data.append(align)
        # every data, it is an array
        # DM_POS = 0
        # DM_PEDICATE = 1
        # DM_SENSE = 2
        # DM_LE = 3
        # DM_CAN_COPY = 4
        # DM_LE_CAN_COPY = 5
        # alignment is in 6
        output_concepts.append(data)

    # if no any potential aligned token found, simply use middle token of the lemma seqs
    # when no any alignments found, it is usually a very shor sentence.
    # TODO: can be improved later for empty nodes
    # find all rest unaligned tokens, which canbe used for remaining unaligned nodes.
    unaligned_index = [i for i in range(n_snt) if i not in aligned_index]  # or [-1 n_snt] for all
    if len(unaligned_index) == 0: unaligned_index = [-1, n_snt]
    #  assert n_snt <= n_top or unaligned_index != [],(n_top,n_snt,concepts,snt_token,top
    # Formally, we add (m−n) NULL concepts to the list. Aligning a word to any NULL, would correspond to saying that the word is not aligned to any ‘real’ concept. After recategorization, m> n for most cases. Then after this, len(concepts) == len(tokens)
    for i in range(n_snt):
        if i in unaligned_index:
            # if token i haven't been used.
            output_concepts.append([NULL_WORD, NULL_WORD, NULL_WORD, NULL_WORD, 0,0, [i]])  # len(top) >= len(snt)

    for i in range(len(output_concepts), n_snt):
        output_concepts.append([NULL_WORD, NULL_WORD, NULL_WORD, NULL_WORD, 0,0, [-1, n_snt]])  # len(top) >= len(snt)

    logged = False
    for i in range(len(output_concepts)):
        if output_concepts[i][-1] == []:
            if not logged:
                logger.info("output_concepts:\n{}\n snt_token, lemma_token, pos:\n{}\nconcepts:\n{}\n top:\n{}".format(output_concepts[i], list(zip(snt_token, lemma_token, pos)), concepts, top))
                logged = True
            # make a default alignments as [-1, n_snt]
            output_concepts[i][-1] = [-1, n_snt]

    rel_feature = []
    rel_tgt = []

    # our model not support empty rels, 0-length packed sequence are not allowed
    # tricks here.
    if root_id == None:
        # there is no root, we choosed from the padded output_concepts
        logger.warn("None root_id for {}".format(top.id))
        root_id = 0

    # rel is an array, every element is a array [[[node, headnodeReCateIndex], [[rel, node2GoldIndex]]]]
    for top_index, role_list in rel:
        top_concept = uni_to_list(top_index[0])  # if align else  uni_to_list(DMUniversal(UNK_WORD,output_concepts[top_index[1]][DM_CAT],NULL_WORD))
        # rel feature, contain the DM_POS, DM_CAT, DM_SENSE, DM_LE, DM_CAN_COPY, not alignments
        # in rel_feature, the top_index is the recategorized index
        rel_feature.append(top_concept[:6] + [top_index[1]])
        #     assert top_index[1] < len(results), (concepts, rel)
        # in rel_tgt, the index for deps are the index from gold top index
        rel_tgt.append(role_list)  # [role,rel_index]

    # output_concepts : [DM_POS, DM_CAT, DM_SENSE, DM_LE, DM_CAN_COPY , aligments]
    # rel: [rel_feature, rel_tgt, root_id], unaligned_inx
    # rel_feature:[[DM_POS, DM_CAT, DM_SENSE, DM_LE, DM_CAN_COPY], the node1index is the index acoording to the order of tanversing the recategorized nodes.
    # rel_tgt: [[rel,node2Index], [rel2, node2index]], the index the gold index
    # root_id is the index in gold_top_index
    # here root_id can be None
    # unaligned_index = [index], this index is the index in the snt
    return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]]

def add_seq_to_dict(dictionary, seq):
    for i in seq:
        dictionary.add(i)


def aligned(align_list):
    return align_list[0] == -1


# id_seq :  [(lemma,cat,lemma_sense,ner])]
def top_seq_to_id(target_pos_dict, cat_dict, sense_dict, lemma_dict, top_seq):
    # pay attention here, for likelihood, it assumes the order of the index is the same with thconcept model returns
    id_seq = []
    for l in top_seq:
        data = [0] * 6
        data[DM_POS] = target_pos_dict[l[DM_POS]]
        data[DM_CAT] = cat_dict[l[DM_CAT]]
        data[DM_SENSE] = sense_dict[l[DM_SENSE]]
        data[DM_LE] = lemma_dict[l[DM_LE]]
        data[DM_CAN_COPY] = l[DM_CAN_COPY]
        data[DM_LE_CAN_COPY] = l[DM_LE_CAN_COPY]
        id_seq.append(data)
    return id_seq


def top_seq_to_dict(target_pos_dict, cat_dict, sense_dict, lemma_dict, top_seq):  # le,cat,le_sense,ner,align
    for i in top_seq:
        target_pos_dict.add(i[DM_POS])
        cat_dict.add(i[DM_CAT])
        sense_dict.add(i[DM_SENSE])
        lemma_dict.add(i[DM_LE])


def rel_seq_to_dict(target_pos_dict, cat_dict, sense_dict, lemma_dict, rel_dict, rel):  # (top,index,[[role,top,index]])
    """
    rel: (rel_feature, rel_tgt, root_id)
    rel_feature: [[POS, CAT, SENSE, LEMMA, node_index]]
    rel_tgt: [[roleStr, node2_index]]
    """
    # relations = [[n, d]], here root_id can be None
    rel_feature, rel_tgt, root_id = rel
    for i in rel_feature:
        target_pos_dict.add(i[DM_POS])
        cat_dict.add(i[DM_CAT])
        sense_dict.add(i[DM_SENSE])
        lemma_dict.add(i[DM_LE])

    # add roleStr into dict
    for role_list in rel_tgt:
        for role_index in role_list:
            #  assert (role_index[0]==":top"),rel_tgt
            rel_dict.add(role_index[0])


def rel_seq_to_id(target_pos_dict, cat_dict, sense_dict, lemma_dict, rel_dict, rel):
    """
    rel: (rel_feature, rel_tgt, root_id)
    rel_feature: [[POS, CAT, SENSE, LEMMA, node1_index]]
    rel_tgt: [[roleStr, node2_index]]
    return
    feature_seq: make rel_features into id:  [ DM_POS, DM_CAT, DM_SNESE=sense, DM_LEMMA]
    index_seq: the gold  index of concept after expansion
    roles_mat: transform rel_tgr into matrix, assuming that only one relation between two node in one direction.
    rootid: a single root_id
    """
    rel_feature, rel_tgt, root_id = rel
    feature_seq = []
    index_seq = []
    roles_mat = []
    # rel feature, contain the DM_CAT, DM_LE, DM_NER, DM_SENSE, index of nodes
    for l in rel_feature:
        data = [0] * 4
        data[0] = target_pos_dict[l[DM_POS]]
        data[1] = cat_dict[l[DM_CAT]]
        data[2] = sense_dict[l[DM_SENSE]]
        data[3] = lemma_dict[l[DM_LE]]
        feature_seq.append(data)
        # top_node index for top relation transdution, this index is recategorized index
        index_seq.append(l[-1])
    for role_list in rel_tgt:
        roles_id = []
        for role_index in role_list:
            # here role_index[0] is still roleStr, role_index[1], the index when tranverse the role of gold top_id
            roles_id.append([role_index[0], role_index[1]])
        roles_mat.append(roles_id)

    return feature_seq, index_seq, roles_mat, root_id

def tok_to_bert_tok(ori_toks, bert_tokenizer):
    ori_to_bert_tok_map = []
    bert_tokens = []
    bert_tokens.append("[CLS]")
    for ori_tok in ori_toks:
        # use the first bert token encoding for the original tok
        ori_to_bert_tok_map.append(len(bert_tokens))
        bert_tokens.extend(bert_tokenizer.tokenize(ori_tok))
    bert_tokens.append("[SEP]")
    return bert_tokens, ori_to_bert_tok_map


def handle_sentence(data, build_dict, n, word_only):
    if n % 1000 == 0:
        logger.info(n)

    input_snt = data["input_snt"]
    snt_token = data["tok"]
    pos = data["pos"]
    lemma_token = data["lem"]
    tok_anchors = data["anchors"]
    if "ner" in data:
        ner = data["ner"]
    else:
        ner = ['O']* len(snt_token)

    if "mwe" in data:
        mwe= data["mwe"]
    else:
        mwe= ['O']* len(snt_token)

    if "top_t" in data:
        top_t = data["top_t"]
    else:
        top_t = None

    # add bert tokens
    # bert_token, ori_to_bert_tok_map = tok_to_bert_tok(snt_token, bert_tokenizer)
    # data['bert_id'] = bert_tokenizer.convert_tokens_to_ids(bert_token)
    #data['tok2bert_index'] = ori_to_bert_tok_map
    data['bert_id'], data['tok2bert_index'] = bert_tokenizer.tokenize(snt_token, split=True, max_tokenized_length = 64)
    data['char_id'] = CharTokenizerUtils.tokenize(snt_token,char_dict)

    if build_dict:
        if word_only:
            # only add tokens into word dict
            add_seq_to_dict(word_dict, snt_token)
        else:
            # add all features into dict, word_dict, lemma_dict, source pos_dict, ner_dict
            add_seq_to_dict(word_dict, snt_token)
            add_seq_to_dict(lemma_dict, lemma_token)
            add_seq_to_dict(source_pos_dict, pos)
            add_seq_to_dict(ner_dict, ner)
            if 'mrp_top' in data:
                top = DMGraph(top_t, data['mrp_top'])
                top_seq, rel, unaligned_index = mytop_to_seq(top, snt_token, lemma_token, pos, tok_anchors, ner, mwe, rl, fragment_to_node_converter)
                top_seq_to_dict(target_pos_dict, cat_dict, sense_dict, lemma_dict, top_seq)
                rel_seq_to_dict(target_pos_dict, cat_dict, sense_dict, lemma_dict, rel_dict, rel)
            else:
                # TODO, for other framworks
                pass
    else:
        # common part for all mrps
        data["snt_id"] = seq_to_id(word_dict, snt_token)[0]
        data["lemma_id"] = seq_to_id(lemma_dict, lemma_token)[0]
        data["pos_id"] = seq_to_id(source_pos_dict, pos)[0]
        data["ner_id"] = seq_to_id(ner_dict, ner)[0]

        l = len(data["pos_id"])
        # If the lengh is not consistent, make an alert
        if not (l == len(data["snt_id"]) and l == len(data["lemma_id"]) and l == len(data["ner_id"])):
            logger.error("length not match: len(data['pos_id'])={}, len(data['snt_id'])={}, len(data['lemma_id'])={}, len(data['ner_id'])={}".format(l, len(data["snt_id"]), len(data["lemma_id"]), len(data["ner_id"])))
            logger.error("data['pos_id]={}".format(data["pos_id"]))
            logger.error("data['snt_id]={}".format(data["snt_id"]))
            logger.error("data['lemma_id']".format(data["lemma_id"]))
            logger.error("data['ner_id]={}".format(data["ner_id"]))
            logger.error("current_pos:{}, snt_token:{}, lemma_token:{}, ner:{}, snt:{}".format(pos, snt_token, lemma_token, ner, data['snt']))
            assert (False)

        # specific for top
        if top_t or 'mrp_top' in data:
            # when not buidling data, make all the tokens and labels in data into id, and save them into pickle-based file.
            top = DMGraph(top_t, data['mrp_top'])
            # output_concepts : [DM_CAT, DM_LE, DM_NER(DM_AUX), DM_SENSE, DM_CAN_COPY, aligments]
            # rel: [rel_feature, rel_tgt, root_id], unaligned_inx
            # rel_feature:[[DM_CAT, DM_LE, DM_NER, DM_SENSE, node1Index], the node1index is the index acoording to the order of tanversing the recategorized nodes.
            # rel_tgt: [[rel,node2Index], [rel2, node2index]], the index the gold index
            # root_id is the index in gold_top_index
            # unaligned_index = [index], this index is the index in the snt
            # return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]
            top_seq, rel, unaligned_index = mytop_to_seq(top, snt_token, lemma_token, pos, tok_anchors, ner, mwe, rl, fragment_to_node_converter)

            data["top_seq"] = top_seq
            # all the top concepts, after recategorizeion. [[subnode1, subnode-attr], [subnode2, subnode]]

            data["top_convertedl_seq"] = top.node_value()
            # return all gold concept and rels,
            # cons: [UniversalDM] all gold concepts
            # roles: [head, depend, relStr]
            data["top_rel_seq"], data["top_rel_triples"] = top.get_gold()
            # the target top ids, include lemma, cat, sense, aux
            data["top_id"] = top_seq_to_id(target_pos_dict, cat_dict, sense_dict, lemma_dict, top_seq)

            data["top_rel_id"], data["top_rel_index"], data["top_roles_mat"], data["top_root"] = rel_seq_to_id(target_pos_dict,
                cat_dict,
                sense_dict,
                lemma_dict,
                rel_dict, rel)

            for i in data["top_rel_index"]:
                assert i < len(data["top_id"]), (data["top_rel_index"], top_seq, data["top_id"])
            # index means the potentially aligned index in the tokenized snt
            # data["index"] [[],[],[]], one inner array for each node in the data, index, the recatgorized id
            data["top_index"] = [all[-1] for all in top_seq]

def buildData(all_data, outPickleFile, build_dict=False, word_only=False):
    """
    i
    """
    n = 0
    # now only handle the top data, to extend for all other formats
    top_data = []
    for id, data in all_data.items():
        n = n + 1
        # try to extend this to other frameworks
        if 'mrp_top' in data:
            # TODO: annoate data
            data = input_preprocessor.annotate_mwe(data)
            handle_sentence(data, build_dict, n, word_only)
            top_data.append(data)
        elif 'test' in outPickleFile:
            # no graph
            data = input_preprocessor.annotate_mwe(data)
            handle_sentence(data, build_dict, n, word_only)
            top_data.append(data)

    logger.info("{} has been preprocessed, {} are good for training".format(n, len(top_data)))

    if n>0:
        bert_len_dict = {0:0, 64:0, 128:0, 256:0}
        for data in top_data:
            l = len(data['bert_id'])
            for i in [0, 64, 128, 256]:
                if l >= i:
                    bert_len_dict[i] +=1
        logger.info("bert_len_dict : {}".format(bert_len_dict))

    if not build_dict:
        outfile = Pickle_Helper(outPickleFile)
        outfile.dump(top_data, "data")
        outfile.save()
    return len(top_data)


# Creating ReUsable Object
rl = DMRules()
input_preprocessor = DMInputPreprocessor(opt, core_nlp_url, top_mwe_file)
non_rule_set = dict()
non_le_rule_set = dict()
rl.load(opt.build_folder+"dicts/top_rule_f")
fragment_to_node_converter = DMReCategorizor(from_file=False, path=opt.build_folder+"dicts/top_recategorization", training=False)

non_rule_set_f = Pickle_Helper(opt.build_folder+"dicts/top_non_rule_set")
non_rule_set = non_rule_set_f.load()["top_non_rule_set"]
threshold = opt.threshold
high_frequency, low_frequency = DMRules.unmixe(non_rule_set, threshold)
# expansion for a dictionary, add both constant and non-constant unaligned nodes
high_freq = {**high_frequency}
logger.info(
    "initial converted,threshold={},len(non_rule_set)={},high_frequency={},low_frequency={},high_freq={}".format(
        threshold, len(non_rule_set), len(high_frequency),
        len(low_frequency), len(high_freq)))

non_le_rule_set_f = Pickle_Helper(opt.build_folder+"dicts/top_non_le_rule_set")
non_le_rule_set = non_le_rule_set_f.load()["top_non_le_rule_set"]
threshold = opt.threshold
high_le_frequency, low_le_frequency = DMRules.unmixe(non_le_rule_set, threshold)
# expansion for a dictionary, add both constant and non-constant unaligned nodes
high_le_freq = {**high_le_frequency}
logger.info(
    "initial converted,threshold={},len(non_le_rule_set)={},high_le_frequency={},low_le_frequency={},high_le_freq={}".format(
        threshold, len(non_le_rule_set), len(high_le_frequency),
        len(low_le_frequency), len(high_le_freq)))



def initial_dict(filename, with_unk=False):
    """
    initial a dict with filename, add special NULL_WORD (""), and unknown OOV word.
    """
    d = Dict(filename)
    # NULL_WORD is always 0, UNK_WORD is 1
    d.addSpecial(NULL_WORD)
    if with_unk:
        d.addSpecial(UNK_WORD)
    #        d.addSpecial(BOS_WORD)
    return d

if not opt.skip:
    # skip means if dict existed, not build it.
    # init all the dict, which can be used as vocabulary of the neual model.
    # used as input source
    if opt.merge_common_dicts:
        try:
            word_dict = Dict(opt.build_folder+"dicts/word_dict")
            word_dict.load()
        except:
            word_dict = initial_dict(opt.build_folder+"dicts/word_dict", with_unk=True)

        try:
            lemma_dict = Dict(opt.build_folder+"dicts/lemma_dict")
            lemma_dict.load()
        except:
            lemma_dict = initial_dict(opt.build_folder+"dicts/lemma_dict", with_unk=True)
        try:
            source_pos_dict = Dict(opt.build_folder+"dicts/pos_dict")
            source_pos_dict.load()
        except:
            source_pos_dict = initial_dict(opt.build_folder+"dicts/pos_dict", with_unk=True)

        try:
            ner_dict = Dict(opt.build_folder+"dicts/ner_dict")
            ner_dict.load()
        except:
            ner_dict = initial_dict(opt.build_folder+"dicts/ner_dict", with_unk=True)  # from stanford

        try:
            char_dict = Dict(opt.build_folder+"dicts/char_dict")
            char_dict.load()
        except:
            char_dict = initial_dict(opt.build_folder+"dicts/char_dict", with_unk=True)  # from stanford
    else:
        word_dict = initial_dict(opt.build_folder+"dicts/top_word_dict", with_unk=True)
        char_dict = initial_dict(opt.build_folder+"dicts/top_char_dict", with_unk=True)
        lemma_dict = initial_dict(opt.build_folder+"dicts/top_lemma_dict", with_unk=True)
        source_pos_dict = initial_dict(opt.build_folder+"dicts/top_source_pos_dict", with_unk=True)
        ner_dict = initial_dict(opt.build_folder+"dicts/top_ner_dict", with_unk=True)  # from stanford

    # for target, pos, cat, sense, rel
    target_pos_dict = initial_dict(opt.build_folder+"dicts/top_target_pos_dict", with_unk=True)

    cat_dict = initial_dict(opt.build_folder+"dicts/top_cat_dict", with_unk=True)

    top_high_dict = initial_dict(opt.build_folder+"dicts/top_high_dict", with_unk=True)
    top_high_le_dict = initial_dict(opt.build_folder+"dicts/top_high_le_dict", with_unk=True)

    rel_dict = initial_dict(opt.build_folder+"dicts/top_rel_dict", with_unk=True)

    sense_dict = initial_dict(opt.build_folder+"dicts/top_sense_dict", with_unk=True)

    for uni in high_freq:
        # for high_dict part, we must keep the same ids
        cat = uni.cat
        id = cat_dict.add(cat)
        top_high_dict.add(cat, id)

    for uni in high_le_freq:
        # for high_dict part, we must keep the same ids
        le = uni.le
        id = lemma_dict.add(le)
        top_high_le_dict.add(le, id)

    # DM, there is no unalgined nodes, all lemma, or target ids can just filled nby the all the nodes in training data
    logger.info("processing training set")
    training_data = readFeaturesInput(trainingCompanionFilesPath)
    mergeWithAnnotatedGraphs(training_data,trainingFilesPath)
    buildData(training_data, trainingPickleFile, build_dict = True, word_only = False)

    logger.info(("processing development set"))
    dev_data = readFeaturesInput(devCompanionFilesPath)
    mergeWithAnnotatedGraphs(dev_data, devFilesPath)
    buildData(dev_data, devPickleFile, build_dict = True, word_only = False)

    logger.info("processing test set")
    test_data = readFeaturesInput(testCompanionFilesPath)
    #mergeWithAnnotatedGraphs(test_data, testFilesPath)
    buildData(test_data, testPickleFile, build_dict = True, word_only = False)

    logger.info("len(word_dict)={}, len(char_dict)={},len(lemma_dict)={}, len(source_pos_dict)={}, len(target_pos_dict)={}, len(ner_dict)={},len(cat_dict)={},len(rel_dict)={},len(sense_dict)={}, threshold={}".format(len(word_dict), len(char_dict),len(lemma_dict), len(source_pos_dict), len(target_pos_dict), len(ner_dict), len(cat_dict), len(rel_dict), len(sense_dict), threshold))
    logger.info("char_dict:\n {}".format(char_dict))
    logger.info("source_pos_dict:\n {}".format(source_pos_dict))
    logger.info("target_pos_dict:\n {}".format(target_pos_dict))
    logger.info("ner_dict:\n {}".format(ner_dict))
    logger.info("cat_dict:\n {}".format(cat_dict))
    logger.info("sense_dict:\n {}".format(sense_dict))
    logger.info("rel_dict:\n {}".format(rel_dict))
    logger.info("high_le_dict:\n {}".format(top_high_le_dict))

    rel_dict = rel_dict.pruneByThreshold(threshold)
    sense_dict = sense_dict.pruneByThreshold(threshold)
    cat_dict = cat_dict.pruneByThreshold(threshold)
    #top_high_dict = gm_high_dict.pruneByThreshold(threshold)

    word_dict.save()
    char_dict.save()
    lemma_dict.save()
    source_pos_dict.save()
    target_pos_dict.save()
    cat_dict.save()
    ner_dict.save()
    rel_dict.save()
    sense_dict.save()
    top_high_dict.save()
    top_high_le_dict.save()

    logger.info("len(word_dict)={}, len(char_dict)={},len(lemma_dict)={}, len(source_pos_dict)={}, len(target_pos_dict)={}, len(ner_dict)={},len(cat_dict)={},len(rel_dict)={},len(sense_dict)={}, threshold={}".format(len(word_dict), len(char_dict),len(lemma_dict), len(source_pos_dict), len(target_pos_dict), len(ner_dict), len(cat_dict), len(rel_dict), len(sense_dict), threshold))
    logger.info("source_pos_dict:\n {}".format(source_pos_dict))
    logger.info("target_pos_dict:\n {}".format(target_pos_dict))
    logger.info("ner_dict:\n {}".format(ner_dict))
    logger.info("cat_dict:\n {}".format(cat_dict))
    logger.info("sense_dict:\n {}".format(sense_dict))
    logger.info("rel_dict:\n {}".format(rel_dict))
    logger.info("top_high_dict:\n {}".format(top_high_dict))
    logger.info("top_high_le_dict:\n {}".format(top_high_le_dict))
else:

    if opt.merge_common_dicts:
        word_dict = Dict(opt.build_folder+"dicts/word_dict")
        char_dict = Dict(opt.build_folder+"dicts/char_dict")
        lemma_dict = Dict(opt.build_folder+"dicts/lemma_dict")
        source_pos_dict = Dict(opt.build_folder+"dicts/pos_dict")
        ner_dict = Dict(opt.build_folder+"dicts/ner_dict")
    else:
        word_dict = Dict(opt.build_folder+"dicts/top_word_dict")
        char_dict = Dict(opt.build_folder+"dicts/top_char_dict")
        lemma_dict = Dict(opt.build_folder+"dicts/top_lemma_dict")
        source_pos_dict = Dict(opt.build_folder+"dicts/top_source_pos_dict")
        ner_dict = Dict(opt.build_folder+"dicts/top_ner_dict")

    target_pos_dict = Dict(opt.build_folder+"dicts/top_target_pos_dict")
    rel_dict = Dict(opt.build_folder+"dicts/top_rel_dict")
    cat_dict = Dict(opt.build_folder+"dicts/top_cat_dict")
    top_high_dict = Dict(opt.build_folder+"dicts/top_high_dict")
    top_high_le_dict = Dict(opt.build_folder+"dicts/top_high_le_dict")
    sense_dict = Dict(opt.build_folder+"dicts/top_sense_dict")

    word_dict.load()
    char_dict.load()
    lemma_dict.load()
    source_pos_dict.load()
    target_pos_dict.load()
    ner_dict.load()
    rel_dict.load()
    cat_dict.load()
    sense_dict.load()
    top_high_dict.load()
    top_high_le_dict.load()

fragment_to_node_converter = DMReCategorizor(from_file=True, path=opt.build_folder+"dicts/top_recategorization", training=False)

logger.info("len(word_dict)={}, len(char_dict)={},len(lemma_dict)={}, len(source_pos_dict)={}, len(target_pos_dict)={}, len(ner_dict)={},len(cat_dict)={},len(rel_dict)={},len(sense_dict)={}, threshold={}".format(len(word_dict), len(char_dict),len(lemma_dict), len(source_pos_dict), len(target_pos_dict), len(ner_dict), len(cat_dict), len(rel_dict), len(sense_dict), threshold))
# after dict has been saved, load them and make the whole dataset into pickle based data point
logger.info("processing training set")
training_data = readFeaturesInput(trainingCompanionFilesPath)
mergeWithAnnotatedGraphs(training_data,trainingFilesPath)
buildData(training_data, trainingPickleFile, build_dict = False, word_only = False)

logger.info(("processing development set"))
dev_data = readFeaturesInput(devCompanionFilesPath)
mergeWithAnnotatedGraphs(dev_data, devFilesPath)
buildData(dev_data, devPickleFile, build_dict = False, word_only = False)

logger.info("processing test set")
test_data = readFeaturesInput(testCompanionFilesPath)
#mergeWithAnnotatedGraphs(test_data, testFilesPath)
buildData(test_data, testPickleFile, build_dict = False, word_only = False)
