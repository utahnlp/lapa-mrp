#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts build dictionary and data into numbers, and seralize into pickle file.

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30

@author: Jie Cao(jiessie.cao@gmail.com)
@since: 2019-06-30
'''

from utility.amr_utils.AMRStringCopyRules import *
from utility.amr_utils.AMRReCategorization import *
from utility.data_helper import *
from parser.Dict import *
import logging
from parser.modules.bert_utils import MRPBertTokenizer
from parser.modules.char_utils import CharTokenizerUtils

import argparse

logger = logging.getLogger("mrp.amr_data_build")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def amr_data_build_parser():
    parser = argparse.ArgumentParser(description='amr_data_build.py')

    ## Data options
    parser.add_argument('--threshold', default=10, type=int,
                        help="""threshold for high frequency concepts""")

    parser.add_argument('--jamr', default=0, type=int,
                        help="""wheather to add .jamr at the end""")
    parser.add_argument('--skip', default=0, type=int,
                        help="""skip dict build if dictionary already built""")
    parser.add_argument('--merge_common_dicts', default=1, type=int,
                        help="""whether to merge common dict if already existed""")
    parser.add_argument('--suffix', default=".mrp_amr", type=str,
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


parser = amr_data_build_parser()
opt = parser.parse_args()
# here use bert to preprare token ids, used for fine tuning.
# if for static bert, we can do that offline, save an encoding for each utterance
bert_tokenizer = MRPBertTokenizer.from_pretrained(opt.bert_model, do_lower_case=opt.do_lower_case)

suffix = opt.suffix + "_jamr" if opt.jamr else opt.suffix
with_jamr = "_with_jamr" if opt.jamr else "_without_jamr"
trainFolderPath = opt.build_folder + "/training/"
trainingPickleFile = opt.build_folder + "/training/training.amr_pickle"+with_jamr+"_processed"
trainingFilesPath = folder_to_files_path(trainFolderPath, suffix)
trainingCompanionFilesPath = folder_to_files_path(trainFolderPath, opt.companion_suffix)

devFolderPath = opt.build_folder + "/dev/"
devPickleFile = opt.build_folder + "/dev/dev.amr_pickle"+with_jamr+"_processed"
devFilesPath = folder_to_files_path(devFolderPath, suffix)
devCompanionFilesPath = folder_to_files_path(devFolderPath, opt.companion_suffix)

testFolderPath = opt.build_folder + "/test/"
testPickleFile = opt.build_folder + "/test/test.amr_pickle"+with_jamr+"_processed"
testFilesPath = folder_to_files_path(testFolderPath, suffix)
testCompanionFilesPath = folder_to_files_path(testFolderPath, opt.companion_suffix)

def myamr_to_seq(amr, snt_token, lemma_token, pos, rl, fragment_to_node_converter,
                 high_freq):  # high_freq should be a dict()
    """
    make a amr to return a squence of node and rels
    # output_concepts : [AMR_CAT, AMR_LE, AMR_NER(AMR_AUX), AMR_SENSE, AMR_CAN_COPY, aligments]
    # rel: [rel_feature, rel_tgt, root_id], unaligned_inx
    # rel_feature:[[AMR_CAT, AMR_LE, AMR_NER, AMR_SENSE, node1Index], the node1index is the index acoording to the order of tanversing the recategorized nodes.
    # rel_tgt: [[rel,node2Index], [rel2, node2index]], the index the gold index
    # root_id is the index in gold_amr_index
    # unaligned_index = [index], this index is the index in the snt
    # return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]
    """

    def uni_to_list(uni, can_copy=0):
        le = uni.le
        cat = uni.cat  # use right category anyway
        ner = uni.aux
        data = [0, 0, 0, 0, 0]
        data[AMR_AUX] = ner
        data[AMR_LE_SENSE] = uni.sense
        data[AMR_LE] = le
        data[AMR_CAT] = cat
        data[AMR_CAN_COPY] = 1 if can_copy else 0
        return data

    output_concepts = []
    lemma_str = " ".join(lemma_token)
    # do recategorization
    fragment_to_node_converter.convert(amr, rl, snt_token, lemma_token, lemma_str)
    # return all concepts, relations, rootid
    # concepts = [[subnode1, subnode-attr], [subnode2, subnode-attr]]
    # rel = [[node, node1ReCateIndex], [[rel, node2GoldIndex]]]], the order is accroding to the index of gold amr index
    # root_id, int, the orderid in all original gold nodes
    concepts, rel, root_id = amr.node_value(keys=["value", "align"], all=True)

    # use rules to match and get all alignments for copy
    # return [[n,c,a], ...], n is variable, c is AMRUniversal, a is alignment[(i,lemma[i],pos[i])]
    results = rl.get_matched_concepts(snt_token, concepts, lemma_token, pos, jamr=opt.jamr)
    aligned_index = []
    n_amr = len(results)
    n_snt = len(snt_token)
    l = len(lemma_token) if lemma_token[-1] != "." else len(lemma_token) - 1

    # hello, linguistic prior here
    old_unaligned_index = [i for i in range(l) if not (
                pos[i] in ["IN", "POS"] or lemma_token[i] == "would" or lemma_token[i] == "will" and pos[i] == "MD"
                or lemma_token[i] == "have" and pos[i] not in ["VB", "VBG"])
                           or lemma_token[i] in ["although", "while", "of", "if", "in", "per", "like", "by", "for"]]

    for i, n_c_a in enumerate(results):
        uni = n_c_a[1]
        # for unaligned node, use old_unaligned_index, a = (i, lemma[i], pos[i])
        # align is [i1, i2 ,i3], n_c_a[2] = a is an arry of possible aligned index[(i,lemma[i],pos[i]), ()], here we only used the first index column
        align = [a[0] for a in n_c_a[2]] if len(n_c_a[2]) > 0 else old_unaligned_index
        # collect all potentially aligned token indices
        # align is (i, lemma[i],pos[i])
        aligned_index += align

        # CAN_COPY is decided by the possible alignment from the string rules, which can find the possible alignments.
        data = uni_to_list(uni, len(n_c_a[2]) > 0)
        # last dimension is alignements, an array of indices can align to the node.
        data.append(align)
        # every data, it is an array
        # AMR_CAT = 0
        # AMR_LE = 1
        # AMR_NER = 2
        # AMR_AUX = 2
        # AMR_LE_SENSE = 3
        # AMR_SENSE = 3
        # AMR_CAN_COPY = 4
        # alignment is in 5
        output_concepts.append(data)

    # if no any potential aligned token found, simply use middle token of the lemma seqs
    # when no any alignments found, it is usually a very shor sentence.
    # TODO: can be improved later.
    if len(aligned_index) == 0:
        output_concepts[0][-1] = [int((len(lemma_token) - 1) / 2)]
        aligned_index = [int((len(lemma_token) - 1) / 2)]
    assert len(aligned_index) > 0, (results, amr._anno, " ".join(lemma_token))
    # find all rest unaligned tokens, which canbe used for remaining unaligned nodes.
    unaligned_index = [i for i in range(n_snt) if i not in aligned_index]  # or [-1 n_snt] for all
    if len(unaligned_index) == 0: unaligned_index = [-1, n_snt]
    #  assert n_snt <= n_amr or unaligned_index != [],(n_amr,n_snt,concepts,snt_token,amr
    # Formally, we add (m−n) NULL concepts to the list. Aligning a word to any NULL, would correspond to saying that the word is not aligned to any ‘real’ concept. After recategorization, m> n for most cases. Then after this, len(concepts) == len(tokens)
    for i in range(n_amr, n_snt):
        output_concepts.append([NULL_WORD, NULL_WORD, NULL_WORD, NULL_WORD, 0, [-1, n_snt]])  # len(amr) >= len(snt)
    logged = False
    for i in range(len(output_concepts)):
        if output_concepts[i][-1] == []:
            if not logged:
                logger.info("output_concepts:\n{}\n snt_token, lemma_token, pos:\n{}\nconcepts:\n{}\n amr:\n{}".format(output_concepts[i], list(zip(snt_token, lemma_token, pos)), concepts, amr))
                logged = True
            # make a default alignments as [-1, n_snt]
            output_concepts[i][-1] = [-1, n_snt]

    rel_feature = []
    rel_tgt = []
    # rel is an array, every element is a array [[[node, headnodeReCateIndex], [[rel, node2GoldIndex]]]]
    for amr_index, role_list in rel:
        amr_concept = uni_to_list(amr_index[
                                      0])  # if align else  uni_to_list(AMRUniversal(UNK_WORD,output_concepts[amr_index[1]][AMR_CAT],NULL_WORD))
        # rel feature, contain the AMR_CAT, AMR_LE, AMR_NER, AMR_SENSE, index of nodes, not alignments
        # in rel_feature, the amr_index is the recategorized index
        rel_feature.append(amr_concept[:4] + [amr_index[1]])
        #     assert amr_index[1] < len(results), (concepts, rel)
        # in rel_tgt, the index for deps are the index from gold amr index
        rel_tgt.append(role_list)  # [role,rel_index]
    # output_concepts : [AMR_CAT, AMR_LE, AMR_NER(AMR_AUX), AMR_SENSE, AMR_CAN_COPY, aligments]
    # rel: [rel_feature, rel_tgt, root_id], unaligned_inx
    # rel_feature:[[AMR_CAT, AMR_LE, AMR_NER, AMR_SENSE, node1Index], the node1index is the index acoording to the order of tanversing the recategorized nodes.
    # rel_tgt: [[rel,node2Index], [rel2, node2index]], the index the gold index
    # root_id is the index in gold_amr_index
    # unaligned_index = [index], this index is the index in the snt
    return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]]


def filter_non_aligned(input_concepts, rel, unaligned_index):
    rel_feature, rel_tgt, root_id = rel

    filtered_index = {}  # original -> filtered

    output_concepts = []
    for i, data in enumerate(input_concepts):
        if len(data[-1]) == 0:
            output_concepts.append(
                [NULL_WORD, NULL_WORD, NULL_WORD, NULL_WORD, 0, unaligned_index])  # len(amr) >= len(snt)
        elif len(data[-1]) == 1 or data[AMR_CAT] == NULL_WORD:
            output_concepts.append(data)
            filtered_index[i] = len(output_concepts) - 1
        else:
            assert False, (i, data, input_concepts, rel)
    out_rel_feature, out_rel_tgt = [], []
    filtered_rel_index = {}  # original -> filtered  for dependency indexing
    for i, data in enumerate(rel_feature):
        index = data[-1]
        if index in filtered_index:
            new_index = filtered_index[index]
            out_rel_feature.append(data[:-1] + [new_index])
            filtered_rel_index[i] = len(out_rel_feature) - 1

    for i, roles in enumerate(rel_tgt):
        if i in filtered_rel_index:
            new_roles = [[role, filtered_rel_index[j]] for role, j in roles if j in filtered_rel_index]
            out_rel_tgt.append(new_roles)

    if root_id not in filtered_rel_index:
        root_id = 0

    assert len(output_concepts) > 0, (input_concepts, rel, unaligned_index)

    return output_concepts, [out_rel_feature, out_rel_tgt, root_id]


def add_seq_to_dict(dictionary, seq):
    for i in seq:
        dictionary.add(i)


def aligned(align_list):
    return align_list[0] == -1


# id_seq :  [(lemma,cat,lemma_sensed,ner])]
def amr_seq_to_id(amr_lemma_dict, amr_category_dict, amr_sense_dict, amr_aux_dict, amr_seq):
    id_seq = []
    for l in amr_seq:
        data = [0] * 5
        data[AMR_CAT] = amr_category_dict[l[AMR_CAT]]
        data[AMR_LE] = amr_lemma_dict[l[AMR_LE]]
        data[AMR_AUX] = amr_aux_dict[l[AMR_AUX]]
        data[AMR_SENSE] = amr_sense_dict[l[AMR_SENSE]]
        data[AMR_CAN_COPY] = l[AMR_CAN_COPY]
        id_seq.append(data)
    return id_seq


def amr_seq_to_dict(amr_lemma_dict, amr_category_dict, amr_sense_dict, amr_aux_dict, amr_seq):  # le,cat,le_sense,ner,align
    for i in amr_seq:
        amr_category_dict.add(i[AMR_CAT])
        amr_lemma_dict.add(i[AMR_LE])
        amr_aux_dict.add(i[AMR_NER])
        amr_sense_dict.add(i[AMR_SENSE])


def rel_seq_to_dict(amr_lemma_dict, amr_category_dict, amr_sense_dict, amr_rel_dict, rel):  # (amr,index,[[role,amr,index]])
    """
    rel: (rel_feature, rel_tgt, root_id)
    rel_feature: [[CAT, LE, NER, SENSE, node_index]]
    rel_tgt: [[roleStr, node2_index]]
    """
    # relations = [[n, d]]
    rel_feature, rel_tgt, root_id = rel
    for i in rel_feature:
        amr_category_dict.add(i[AMR_CAT])
        amr_lemma_dict.add(i[AMR_LE])
    #      amr_sense_dict.add(i[AMR_SENSE])

    # add roleStr into dict
    for role_list in rel_tgt:
        for role_index in role_list:
            #  assert (role_index[0]==":top"),rel_tgt
            amr_rel_dict.add(role_index[0])


def rel_seq_to_id(amr_lemma_dict, amr_category_dict, amr_sense_dict, amr_rel_dict, rel):
    """
    rel: (rel_feature, rel_tgt, root_id)
    rel_feature: [[CAT, LE, NER, SENSE, node1_index]]
    rel_tgt: [[roleStr, node2_index]]
    return 
    feature_seq: make rel_features into id:  [ AMR_CAT=cat, AMR_LE=lemma, AMR_SNESE=sense]
    index_seq: the gold  index of concept after expansion
    roles_mat: transform rel_tgr into matrix, assuming that only one relation between two node in one direction.
    rootid: a single root_id
    """
    rel_feature, rel_tgt, root_id = rel
    feature_seq = []
    index_seq = []
    roles_mat = []
    # rel feature, contain the AMR_CAT, AMR_LE, AMR_NER, AMR_SENSE, index of nodes
    for l in rel_feature:
        data = [0] * 3
        data[0] = amr_category_dict[l[AMR_CAT]]
        data[1] = amr_lemma_dict[l[AMR_LE]]
        data[2] = amr_sense_dict[l[AMR_SENSE]]
        feature_seq.append(data)
        # amr_node index for amr relation transdution, this index is recategorized index
        index_seq.append(l[-1])
    for role_list in rel_tgt:
        roles_id = []
        for role_index in role_list:
            # here role_index[0] is still roleStr, role_index[1], the index when tranverse the role of gold amr_id
            roles_id.append([role_index[0], role_index[1]])
        roles_mat.append(roles_id)

    return feature_seq, index_seq, roles_mat, root_id

def tok_to_bert_tok(ori_toks, bert_tokenizer):
    """
    use bert to tokenize and store the first subword as its token
    """
    ori_to_bert_tok_map = []
    bert_tokens = []
    bert_tokens.append("[CLS]")
    for ori_tok in ori_toks:
        # use the first bert token encoding for the original tok
        # here we only keep the first subword index for that token,
        # [batch_size, num-tokens]
        # it can be recovered to [batch_size, num_tokens, num_sub_words]
        ori_to_bert_tok_map.append(len(bert_tokens))
        bert_tokens.extend(bert_tokenizer.tokenize(ori_tok))
    bert_tokens.append("[SEP]")
    return bert_tokens, ori_to_bert_tok_map

def handle_sentence(data, build_dict, n, word_only):
    if n % 1000 == 0:
        logger.info(n)

    snt_token = data["tok"]
    pos = data["pos"]
    lemma_token = data["lem"]
    if "ner" in data:
        ner = data["ner"]
    else:
        ner = ['O']* len(snt_token)

    if "mwe" in data:
        mwe= data["mwe"]
    else:
        mwe= ['O']* len(snt_token)
        data['mwe'] = mwe

    if "amr_t" in data:
        amr_t = data["amr_t"]
    else:
        amr_t = None

    # add bert tokens
    #bert_token, ori_to_bert_tok_map = tok_to_bert_tok(snt_token, bert_tokenizer)
    #data['bert_id'] = bert_tokenizer.convert_tokens_to_ids(bert_token)
    #data['tok2bert_index'] = ori_to_bert_tok_map
    data['bert_id'], data['tok2bert_index'] = bert_tokenizer.tokenize(snt_token, split=True, max_tokenized_length = 64)
    data['char_id'] = CharTokenizerUtils.tokenize(snt_token,amr_char_dict)

    if build_dict:
        if word_only:
            # only add tokens into word dict
            add_seq_to_dict(amr_word_dict, snt_token)
        else:
            # add all features into dict, amr_word_dict, amr_lemma_dict, amr_pos_dict, amr_ner_dict
            add_seq_to_dict(amr_word_dict, snt_token)
            add_seq_to_dict(amr_lemma_dict, lemma_token)
            add_seq_to_dict(amr_pos_dict, pos)
            add_seq_to_dict(amr_ner_dict, ner)
            if 'mrp_amr' in data:
                amr = AMRGraph(amr_t, data['mrp_amr'])
                amr_seq, rel, unaligned_index = myamr_to_seq(amr, snt_token, lemma_token, pos, rl, fragment_to_node_converter, high_freq)
                amr_seq_to_dict(amr_lemma_dict, amr_category_dict, amr_sense_dict, amr_aux_dict, amr_seq)
                rel_seq_to_dict(amr_lemma_dict, amr_category_dict, amr_sense_dict, amr_rel_dict, rel)
            else:
                # TODO, for other framworks
                pass
    else:
        # common part for all mrps
        data["snt_id"] = seq_to_id(amr_word_dict, snt_token)[0]
        data["lemma_id"] = seq_to_id(amr_lemma_dict, lemma_token)[0]
        data["pos_id"] = seq_to_id(amr_pos_dict, pos)[0]
        data["ner_id"] = seq_to_id(amr_ner_dict, ner)[0]

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

        # specific for amr
        if amr_t or 'mrp_amr' in data:
            # when not buidling data, make all the tokens and labels in data into id, and save them into pickle-based file.
            amr = AMRGraph(amr_t, data['mrp_amr'])
            # output_concepts : [AMR_CAT, AMR_LE, AMR_NER(AMR_AUX), AMR_SENSE, AMR_CAN_COPY, aligments]
            # rel: [rel_feature, rel_tgt, root_id], unaligned_inx
            # rel_feature:[[AMR_CAT, AMR_LE, AMR_NER, AMR_SENSE, node1Index], the node1index is the index acoording to the order of tanversing the recategorized nodes.
            # rel_tgt: [[rel,node2Index], [rel2, node2index]], the index the gold index
            # root_id is the index in gold_amr_index
            # unaligned_index = [index], this index is the index in the snt
            # return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]
            amr_seq, rel, unaligned_index = myamr_to_seq(amr, snt_token, lemma_token, pos, rl, fragment_to_node_converter,
                                                         high_freq)
            if opt.jamr:
                amr_seq, rel = filter_non_aligned(amr_seq, rel, unaligned_index)

            data["amr_seq"] = amr_seq
            # all the amr concepts, after recategorizeion. [[subnode1, subnode-attr], [subnode2, subnode]]

            data["amr_convertedl_seq"] = amr.node_value()
            # return all gold concept and rels,
            # cons: [UniversalAMR] all gold concepts
            # roles: [head, depend, relStr]
            data["amr_rel_seq"], data["amr_rel_triples"] = amr.get_gold()
            # the target amr ids, include lemma, cat, sense, aux
            data["amr_id"] = amr_seq_to_id(amr_lemma_dict, amr_category_dict, amr_sense_dict, amr_aux_dict, amr_seq)

            data["amr_rel_id"], data["amr_rel_index"], data["amr_roles_mat"], data["amr_root"] = rel_seq_to_id(amr_lemma_dict,
                                                                                                   amr_category_dict,
                                                                                                   amr_sense_dict,
                                                                                                   amr_rel_dict, rel)

            for i in data["amr_rel_index"]:
                assert i < len(data["amr_id"]), (data["amr_rel_index"], amr_seq, data["amr_id"])
            # index means the potentially aligned index in the tokenized snt
            # data["index"] [[],[],[]], one inner array for each node in the data, index, the recatgorized id
            data["amr_index"] = [all[-1] for all in amr_seq]

def buildData(all_data, outPickleFile, build_dict=False, word_only=False):
    """
    i
    """
    n = 0
    # now only handle the amr data, to extend for all other formats
    amr_data = []
    for id, data in all_data.items():
        # try to extend this to other frameworks
        if 'mrp_amr' in data:
            n = n + 1
            handle_sentence(data, build_dict, n, word_only)
            amr_data.append(data)
        elif 'test' in outPickleFile:
            handle_sentence(data, build_dict, n, word_only)
            amr_data.append(data)

    if n>0:
        bert_len_dict = {0:0, 64:0, 128:0, 256:0}
        for data in amr_data:
            l = len(data['bert_id'])
            for i in [0, 64, 128, 256]:
                if l >= i:
                    bert_len_dict[i] +=1
        logger.info("bert_len_dict : {}".format(bert_len_dict))

    if not build_dict:
        outfile = Pickle_Helper(outPickleFile)
        outfile.dump(amr_data, "data")
        outfile.save()
    return len(amr_data)


# Creating ReUsable Object
rl = AMRRules()
rl.load(opt.build_folder+"dicts/amr_rule_f" + with_jamr)
# initializer = lasagne.init.Uniform()
fragment_to_node_converter = AMRReCategorizor(from_file=False, path=opt.build_folder+"dicts/graph_to_node_dict_extended" + with_jamr, training=False, auto_convert_threshold=opt.threshold)
non_rule_set_f = Pickle_Helper(opt.build_folder+"dicts/amr_non_rule_set")
non_rule_set = non_rule_set_f.load()["amr_non_rule_set"]
threshold = opt.threshold
high_text_num, high_frequency, low_frequency, low_text_num = unmixe(non_rule_set, threshold)
# expansion for a dictionary, add both constant and non-constant unaligned nodes
high_freq = {**high_text_num, **high_frequency}
logger.info(
    "initial converted,threshold={},len(non_rule_set)={},high_text_num={},high_frequency={},low_frequency={},low_text_num={},high_freq={}".format(
        threshold, len(non_rule_set), len(high_text_num), len(high_frequency),
        len(low_frequency), len(low_text_num), len(high_freq)))


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
    if opt.merge_common_dicts:
        try:
            amr_word_dict = Dict(opt.build_folder+"dicts/word_dict")
            amr_word_dict.load()
        except:
            amr_word_dict = initial_dict(opt.build_folder+"dicts/word_dict", with_unk=True)

        try:
            amr_lemma_dict = Dict(opt.build_folder+"dicts/lemma_dict")
            amr_lemma_dict.load()
        except:
            amr_lemma_dict = initial_dict(opt.build_folder+"dicts/lemma_dict", with_unk=True)

        try:
            amr_pos_dict = Dict(opt.build_folder+"dicts/pos_dict")
            amr_pos_dict.load()
        except:
            amr_pos_dict = initial_dict(opt.build_folder+"dicts/pos_dict", with_unk=True)

        try:
            amr_ner_dict = Dict(opt.build_folder+"dicts/ner_dict")
            amr_ner_dict.load()
        except:
            amr_ner_dict = initial_dict(opt.build_folder+"dicts/ner_dict", with_unk=True)  # from stanford

        try:
            amr_char_dict = Dict(opt.build_folder+"dicts/char_dict")
            amr_char_dict.load()
        except:
            amr_char_dict = initial_dict(opt.build_folder+"dicts/char_dict", with_unk=True)  # from stanford
    else:
        amr_word_dict = initial_dict(opt.build_folder+"dicts/word_dict", with_unk=True)
        amr_char_dict = initial_dict(opt.build_folder+"dicts/char_dict", with_unk=True)
        amr_lemma_dict = initial_dict(opt.build_folder+"dicts/lemma_dict", with_unk=True)
        amr_pos_dict = initial_dict(opt.build_folder+"dicts/pos_dict", with_unk=True)
        amr_ner_dict = initial_dict(opt.build_folder+"dicts/ner_dict", with_unk=True)  # from stanford

    amr_high_dict = initial_dict(opt.build_folder+"dicts/amr_high_dict", with_unk=True)
    amr_aux_dict = initial_dict(opt.build_folder+"dicts/amr_aux_dict", with_unk=True)

    amr_rel_dict = initial_dict(opt.build_folder+"dicts/amr_rel_dict", with_unk=True)

    amr_category_dict = initial_dict(opt.build_folder+"dicts/amr_category_dict", with_unk=True)

    amr_sense_dict = initial_dict(opt.build_folder+"dicts/amr_sense_dict", with_unk=True)

    # prepare amr_high_dict and amr_lemma_dict
    for uni in high_freq:
        le = uni.le
        amr_lemma_dict.add(le)
        amr_high_dict.add(le)


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

    logger.info("len(amr_aux_dict)={},len(amr_rel_dict)={},threshold={}".format(len(amr_aux_dict), len(amr_rel_dict), threshold))
    logger.info("amr_aux_dict:\n {}".format(amr_aux_dict))
    logger.info("amr_rel_dict:\n {}".format(amr_rel_dict))

    amr_rel_dict = amr_rel_dict.pruneByThreshold(threshold)
    amr_aux_dict = amr_aux_dict.pruneByThreshold(threshold)
    amr_category_dict = amr_category_dict.pruneByThreshold(threshold)
    amr_word_dict.save()
    amr_char_dict.save()
    amr_lemma_dict.save()
    amr_pos_dict.save()
    amr_aux_dict.save()
    amr_ner_dict.save()
    amr_high_dict.save()
    amr_category_dict.save()
    amr_rel_dict.save()
    amr_sense_dict.save()
    logger.info("len(amr_aux_dict)={},len(amr_rel_dict)={},threshold={}".format(len(amr_aux_dict), len(amr_rel_dict), threshold))
    logger.info("amr_aux_dict:\n {}".format(amr_aux_dict))
    logger.info("amr_rel_dict:\n {}".format(amr_rel_dict))
else:

    if opt.merge_common_dicts:
        amr_word_dict = Dict(opt.build_folder+"dicts/word_dict")
        amr_char_dict = Dict(opt.build_folder+"dicts/char_dict")
        amr_lemma_dict = Dict(opt.build_folder+"dicts/lemma_dict")
        amr_pos_dict = Dict(opt.build_folder+"dicts/pos_dict")
        amr_ner_dict = Dict(opt.build_folder+"dicts/ner_dict")
    else:
        amr_word_dict = Dict(opt.build_folder+"dicts/word_dict")
        amr_char_dict = Dict(opt.build_folder+"dicts/char_dict")
        amr_lemma_dict = Dict(opt.build_folder+"dicts/lemma_dict")
        amr_pos_dict = Dict(opt.build_folder+"dicts/pos_dict")
        amr_ner_dict = Dict(opt.build_folder+"dicts/ner_dict")

    amr_aux_dict = Dict(opt.build_folder+"dicts/amr_aux_dict")
    amr_high_dict = Dict(opt.build_folder+"dicts/amr_high_dict")
    amr_rel_dict = Dict(opt.build_folder+"dicts/amr_rel_dict")
    amr_category_dict = Dict(opt.build_folder+"dicts/amr_category_dict")
    amr_sense_dict = Dict(opt.build_folder+"dicts/amr_sense_dict")

    amr_word_dict.load()
    amr_char_dict.load()
    amr_lemma_dict.load()
    amr_pos_dict.load()
    amr_ner_dict.load()
    amr_rel_dict.load()
    amr_category_dict.load()
    amr_high_dict.load()
    amr_aux_dict.load()
    amr_sense_dict.save()

fragment_to_node_converter = AMRReCategorizor(from_file=True, path=opt.build_folder+"dicts/graph_to_node_dict_extended" + with_jamr,
                                           training=False, ner_cat_dict=amr_aux_dict)
logger.info("dictionary building done")
logger.info("amr_word_dict={}\t amr_char_dict={} \t amr_lemma_dict={}\t amr_pos_dict={}\t amr_ner_dict={}\t amr_high_dict={}\t amr_sense_dict={}\t amr_category_dict={}\t amr_aux_dict={}\t amr_rel_dict={}".format(
    len(amr_word_dict), len(amr_char_dict), len(amr_lemma_dict), len(amr_pos_dict), len(amr_ner_dict),
    len(amr_high_dict), len(amr_sense_dict), len(amr_category_dict), len(amr_aux_dict), len(amr_rel_dict)))

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

logger.info("initial converted,threshold={},len(non_rule_set)={},high_text_num={},high_frequency={},low_frequency={},low_text_num={}".format(
    threshold, len(non_rule_set), len(high_text_num), len(high_frequency),
    len(low_frequency), len(low_text_num)))

