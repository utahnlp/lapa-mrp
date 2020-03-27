#!/usr/bin/env python3.6
# coding=utf-8
'''

Scripts build dictionary and data into numbers, and seralize into pickle file.

Data path information should also be specified here for
trainFolderPath, devFolderPath and testFolderPath
as we allow option to choose from two version of data.

@author: Chunchuan Lyu (chunchuan.lv@gmail.com)
@since: 2018-05-30
'''

from utility.psd_utils.PSDStringCopyRules import *
from utility.psd_utils.PSDReCategorization import *
from utility.data_helper import *
from parser.Dict import *
from parser.PSDProcessors import *
from utility.constants import *
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

def psd_data_build_parser():
    parser = argparse.ArgumentParser(description='psd_data_build.py')

    ## Data options
    parser.add_argument('--threshold', default=10, type=int,
                        help="""threshold for high frequency concepts""")

    parser.add_argument('--skip', default=0, type=int,
                        help="""skip dict build if dictionary already built""")
    parser.add_argument('--merge_common_dicts', default=1, type=int,
                        help="""whether to merge common dict if already existed""")
    parser.add_argument('--suffix', default=".mrp_psd", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--companion_suffix', default=".mrp_conllu_pre_processed", type=str,
                        help="""suffix of files to combine""")
    parser.add_argument('--build_folder', default="", type=str,
                        help="""the build folder for dict and rules, data""")
    parser.add_argument('--bert_model', default="bert-base-cased", type=str,
                        help="""the name for pretraind bert name""")
    parser.add_argument('--do_lower_case', default=False, type=bool,
                        help="""Whether to lower case when tokenization""")
    return parser


parser = psd_data_build_parser()
opt = parser.parse_args()
# here use bert to preprare token ids, used for fine tuning.
# if for static bert, we can do that offline, save an encoding for each utterance
bert_tokenizer = MRPBertTokenizer.from_pretrained(opt.bert_model, do_lower_case=opt.do_lower_case)

suffix = opt.suffix
trainFolderPath = opt.build_folder + "/training/"
trainingPickleFile = opt.build_folder + "/training/training.psd_pickle_processed"
trainingFilesPath = folder_to_files_path(trainFolderPath, suffix)
trainingCompanionFilesPath = folder_to_files_path(trainFolderPath, opt.companion_suffix)

devFolderPath = opt.build_folder + "/dev/"
devPickleFile = opt.build_folder + "/dev/dev.psd_pickle_processed"
devFilesPath = folder_to_files_path(devFolderPath, suffix)
devCompanionFilesPath = folder_to_files_path(devFolderPath, opt.companion_suffix)

testFolderPath = opt.build_folder + "/test/"
testPickleFile = opt.build_folder + "/test/test.psd_pickle_processed"
testFilesPath = folder_to_files_path(testFolderPath, suffix)
testCompanionFilesPath = folder_to_files_path(testFolderPath, opt.companion_suffix)

def mypsd_to_seq(psd, snt_token, lemma_token, pos, mwe_token, tok_anchors, rl, fragment_to_node_converter):  # high_freq should be a dict()
    """
    make a psd to return a squence of node and rels
    # output_concepts : [PSD_CAT, PSD_LE, PSD_NER(PSD_AUX), PSD_SENSE, PSD_CAN_COPY, aligments]
    # rel: [rel_feature, rel_tgt, root_id], unaligned_inx
    # rel_feature:[[PSD_CAT, PSD_LE, PSD_NER, PSD_SENSE, node1Index], the node1index is the index acoording to the order of tanversing the recategorized nodes.
    # rel_tgt: [[rel,node2Index], [rel2, node2index]], the index the gold index
    # root_id is the index in gold_psd_index
    # unaligned_index = [index], this index is the index in the snt
    # return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]
    """

    def uni_to_list(uni, copy_from=0):
        le = uni.le
        pos = uni.pos
        sense = uni.sense
        data = [0, 0, 0, 0]
        data[PSD_POS] = pos
        data[PSD_SENSE] = sense
        data[PSD_LE] = le
        # 0 means not copy, 1 means copy from word, 2 means copy from lemma
        data[PSD_CAN_COPY] = copy_from
        return data

    output_concepts = []
    lemma_str = " ".join(lemma_token)
    # do align before converting., add align
    rl.get_matched_concepts(snt_token,psd,lemma_token, pos, mwe_token, tok_anchors)

    # do recategorization
    fragment_to_node_converter.convert(psd, rl, snt_token, lemma_token, pos, lemma_token, mwe_token)
    # return all concepts, relations, rootid
    # concepts = [[subnode1, subnode-attr], [subnode2, subnode-attr]]
    # rel = [[node, node1ReCateIndex], [[rel, node2GoldIndex]]]], the order is accroding to the index of gold psd index
    # root_id, int, the orderid in all original gold nodes
    concepts, rel, root_id = psd.node_value(keys=["value", "align"], all=True)

    results = rl.get_matched_concepts(snt_token,psd,lemma_token, pos,mwe_token, tok_anchors)
    # use rules to match and get all alignments for copy
    # return [[n,c,a], ...], n is variable, c is PSDUniversal, a is alignment[(i,lemma[i],pos[i])]
    aligned_index = []
    n_psd = len(concepts)
    n_snt = len(snt_token)
    l = len(lemma_token) if lemma_token[-1] != "." else len(lemma_token) - 1

    results = rl.get_matched_concepts(snt_token,psd,lemma_token, pos,mwe_token,tok_anchors)
    for i, n_c_a_f in enumerate(results):
        uni = n_c_a_f[1]
        align = n_c_a_f[2]
        aligned_index += align

        data = uni_to_list(uni, n_c_a_f[3])
        # last dimension is alignements, an array of indices can align to the node.
        data.append(align)
        # every data, it is an array
        # PSD_POS = 0
        # PSD_SENSE = 1
        # PSD_LE = 2
        # PSD_CAN_COPY = 3
        # alignment is in 4
        output_concepts.append(data)

    # if no any potential aligned token found, simply use middle token of the lemma seqs
    # when no any alignments found, it is usually a very shor sentence.
    # TODO: can be improved later for empty nodes
    # find all rest unaligned tokens, which canbe used for remaining unaligned nodes.
    unaligned_index = [i for i in range(n_snt) if i not in aligned_index]  # or [-1 n_snt] for all
    if len(unaligned_index) == 0: unaligned_index = [-1, n_snt]
    #  assert n_snt <= n_psd or unaligned_index != [],(n_psd,n_snt,concepts,snt_token,psd
    # Formally, we add (m−n) NULL concepts to the list. Aligning a word to any NULL, would correspond to saying that the word is not aligned to any ‘real’ concept. After recategorization, m> n for most cases. Then after this, len(concepts) == len(tokens)

    for i in range(n_snt):
        if i in unaligned_index:
            # if token i haven't been used.
            output_concepts.append([NULL_WORD, NULL_WORD, NULL_WORD, 0, [i]])  # len(dm) >= len(snt)

    for i in range(n_psd, n_snt):
        output_concepts.append([NULL_WORD, NULL_WORD, NULL_WORD, 0, [-1, n_snt]])  # len(psd) >= len(snt)

    logged = False
    for i in range(len(output_concepts)):
        if output_concepts[i][-1] == []:
            if not logged:
                logger.info("output_concepts:\n{}\n snt_token, lemma_token, pos:\n{}\nconcepts:\n{}\n psd:\n{}".format(output_concepts[i], list(zip(snt_token, lemma_token, pos)), concepts, psd))
                logged = True
            # make a default alignments as [-1, n_snt]
            output_concepts[i][-1] = [-1, n_snt]

    rel_feature = []
    rel_tgt = []

    # our model not support empty rels, 0-length packed sequence are not allowed
    # tricks here.
    if root_id == None:
        # there is no root, we choosed from the padded output_concepts
        logger.warn("None root_id for {}".format(psd.id))
        root_id = 0

    # rel is an array, every element is a array [[[node, headnodeReCateIndex], [[rel, node2GoldIndex]]]]
    for psd_index, role_list in rel:
        psd_concept = uni_to_list(psd_index[0])  # if align else  uni_to_list(PSDUniversal(UNK_WORD,output_concepts[psd_index[1]][PSD_CAT],NULL_WORD))
        # rel feature, contain the PSD_POS, PSD_LE, PSD_SENSE, PSD_CAN_COPY, not alignments
        # in rel_feature, the psd_index is the recategorized index
        rel_feature.append(psd_concept[:4] + [psd_index[1]])
        #     assert psd_index[1] < len(results), (concepts, rel)
        # in rel_tgt, the index for deps are the index from gold psd index
        rel_tgt.append(role_list)  # [role,rel_index]

    # output_concepts : [PSD_POS, PSD_LE, PSD_SENSE, PSD_CAN_COPY , aligments]
    # rel: [rel_feature, rel_tgt, root_id], unaligned_inx
    # rel_feature:[[PSD_POS, PSD_LE, PSD_SENSE, PSD_CAN_COPY], the node1index is the index acoording to the order of tanversing the recategorized nodes.
    # rel_tgt: [[rel,node2Index], [rel2, node2index]], the index the gold index
    # root_id is the index in gold_psd_index
    # here root_id can be None
    # unaligned_index = [index], this index is the index in the snt
    return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]]


def add_seq_to_dict(dictionary, seq):
    for i in seq:
        dictionary.add(i)

def aligned(align_list):
    return align_list[0] == -1


# id_seq :  [(lemma,cat,lemma_sense,ner])]
def psd_seq_to_id(target_pos_dict, sense_dict, lemma_dict, psd_seq):
    # pay attention here, for likelihood, it assumes the order of the index is the same with thconcept model returns
    id_seq = []
    for l in psd_seq:
        data = [0] * 4
        data[PSD_POS] = target_pos_dict[l[PSD_POS]]
        data[PSD_LE] = lemma_dict[l[PSD_LE]]
        data[PSD_SENSE] = sense_dict[l[PSD_SENSE]]
        data[PSD_CAN_COPY] = l[PSD_CAN_COPY]
        id_seq.append(data)
    return id_seq


def psd_seq_to_dict(target_pos_dict, sense_dict, lemma_dict, psd_seq):  # le,cat,le_sense,ner,align
    for i in psd_seq:
        target_pos_dict.add(i[PSD_POS])
        lemma_dict.add(i[PSD_LE])
        sense_dict.add(i[PSD_SENSE])


def rel_seq_to_dict(target_pos_dict, sense_dict, lemma_dict, rel_dict, rel):  # (psd,index,[[role,psd,index]])
    """
    rel: (rel_feature, rel_tgt, root_id)
    rel_feature: [[POS, LEMMA, SENSE, node_index]]
    rel_tgt: [[roleStr, node2_index]]
    """
    # relations = [[n, d]], here root_id can be None
    rel_feature, rel_tgt, root_id = rel
    for i in rel_feature:
        target_pos_dict.add(i[PSD_POS])
        lemma_dict.add(i[PSD_LE])
        sense_dict.add(i[PSD_SENSE])

    # add roleStr into dict
    for role_list in rel_tgt:
        for role_index in role_list:
            #  assert (role_index[0]==":top"),rel_tgt
            rel_dict.add(role_index[0])


def rel_seq_to_id(target_pos_dict, sense_dict, lemma_dict, rel_dict, rel):
    """
    rel: (rel_feature, rel_tgt, root_id)
    rel_feature: [[POS, LEMMA, SENSE, node1_index]]
    rel_tgt: [[roleStr, node2_index]]
    return
    feature_seq: make rel_features into id:  [ PSD_POS, PSD_SNESE=sense, PSD_LE]
    index_seq: the gold  index of concept after expansion
    roles_mat: transform rel_tgr into matrix, assuming that only one relation between two node in one direction.
    rootid: a single root_id
    """
    rel_feature, rel_tgt, root_id = rel
    feature_seq = []
    index_seq = []
    roles_mat = []
    # rel feature, contain the PSD_CAT, PSD_LE, PSD_NER, PSD_SENSE, index of nodes
    for l in rel_feature:
        data = [0] * 3
        data[PSD_POS] = target_pos_dict[l[PSD_POS]]
        data[PSD_LE] = lemma_dict[l[PSD_LE]]
        data[PSD_SENSE] = sense_dict[l[PSD_SENSE]]
        feature_seq.append(data)
        # psd_node index for psd relation transdution, this index is recategorized index
        index_seq.append(l[-1])
    for role_list in rel_tgt:
        roles_id = []
        for role_index in role_list:
            # here role_index[0] is still roleStr, role_index[1], the index when tranverse the role of gold psd_id
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
    snt_token = data["tok"]
    pos = data["pos"]
    lemma_token = data["lem"]
    tok_anchors = data["anchors"]
    if "ner" in data:
        ner = data["ner"]
    else:
        ner = ['O']* len(snt_token)

    if "mwe" in data:
        mwe_token= data["mwe"]
    else:
        mwe_token= ['O']* len(snt_token)

    if "psd_t" in data:
        psd_t = data["psd_t"]
    else:
        psd_t = None

    # add bert tokensk
    #bert_token, ori_to_bert_tok_map = tok_to_bert_tok(snt_token, bert_tokenizer)
    #data['bert_id'] = bert_tokenizer.convert_tokens_to_ids(bert_token)
    #data['tok2bert_index'] = ori_to_bert_tok_map

    data['bert_id'], data['tok2bert_index'] = bert_tokenizer.tokenize(snt_token, split=True, max_tokenized_length = 64)

    data['char_id'] = CharTokenizerUtils.tokenize(snt_token,char_dict)

    if build_dict:
        if word_only:
            # only add tokens into word dict
            add_seq_to_dict(word_dict,lemma_dict,snt_token)
        else:
            # add all features into dict, word_dict, lemma_dict, source pos_dict, ner_dict
            add_seq_to_dict(word_dict,snt_token)
            add_seq_to_dict(lemma_dict,lemma_token)
            add_seq_to_dict(source_pos_dict, pos)
            add_seq_to_dict(ner_dict, ner)
            if 'mrp_psd' in data:
                psd = PSDGraph(psd_t, data['mrp_psd'])
                psd_seq, rel, unaligned_index = mypsd_to_seq(psd, snt_token, lemma_token, pos,mwe_token, tok_anchors, rl, fragment_to_node_converter)
                psd_seq_to_dict(target_pos_dict, sense_dict, lemma_dict, psd_seq)
                rel_seq_to_dict(target_pos_dict, sense_dict, lemma_dict, rel_dict, rel)
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

        # specific for psd
        if psd_t or 'mrp_psd' in data:
            # when not buidling data, make all the tokens and labels in data into id, and save them into pickle-based file.
            psd = PSDGraph(psd_t, data['mrp_psd'])
            # rel: [rel_feature, rel_tgt, root_id], unaligned_inx
            # rel_feature:[[PSD_CAT, PSD_LE, PSD_NER, PSD_SENSE, node1Index], the node1index is the index acoording to the order of tanversing the recategorized nodes.
            # rel_tgt: [[rel,node2Index], [rel2, node2index]], the index the gold index
            # root_id is the index in gold_psd_index
            # unaligned_index = [index], this index is the index in the snt
            # return output_concepts, [rel_feature, rel_tgt, root_id], unaligned_index  # [[[lemma1,lemma2],category,relation]
            psd_seq, rel, unaligned_index = mypsd_to_seq(psd, snt_token, lemma_token, pos, mwe_token, tok_anchors, rl, fragment_to_node_converter)

            data["psd_seq"] = psd_seq
            # all the psd concepts, after recategorizeion. [[subnode1, subnode-attr], [subnode2, subnode]]

            data["psd_convertedl_seq"] = psd.node_value()
            # return all gold concept and rels,
            # cons: [UniversalPSD] all gold concepts
            # roles: [head, depend, relStr]
            data["psd_rel_seq"], data["psd_rel_triples"] = psd.get_gold()
            # the target psd ids, include lemma, cat, sense, aux
            data["psd_id"] = psd_seq_to_id(target_pos_dict, sense_dict, lemma_dict, psd_seq)

            data["psd_rel_id"], data["psd_rel_index"], data["psd_roles_mat"], data["psd_root"] = rel_seq_to_id(target_pos_dict,
                sense_dict,
                lemma_dict,
                rel_dict, rel)

            for i in data["psd_rel_index"]:
                assert i < len(data["psd_id"]), (data["psd_rel_index"], psd_seq, data["psd_id"])
            # index means the potentially aligned index in the tokenized snt
            # data["index"] [[],[],[]], one inner array for each node in the data, index, the recatgorized id
            data["psd_index"] = [all[-1] for all in psd_seq]

def buildData(all_data, outPickleFile, build_dict=False, word_only=False):
    """
    i
    """
    n = 0
    # now only handle the psd data, to extend for all other formats
    psd_data = []
    for id, data in all_data.items():
        # try to extend this to other frameworks
        if 'mrp_psd' in data:
            n = n + 1
            data = input_preprocessor.annotate_mwe(data)
            handle_sentence(data, build_dict, n, word_only)
            psd_data.append(data)
        elif 'test' in outPickleFile:
            # no graph
            data = input_preprocessor.annotate_mwe(data)
            handle_sentence(data, build_dict, n, word_only)
            psd_data.append(data)


    logger.info("{} has been preprocessed, {} are good for training".format(n, len(psd_data)))

    if n>0:
        bert_len_dict = {0:0, 64:0, 128:0, 256:0}
        for data in psd_data:
            l = len(data['bert_id'])
            for i in [0, 64, 128, 256]:
                if l >= i:
                    bert_len_dict[i] +=1
        logger.info("bert_len_dict : {}".format(bert_len_dict))

    if not build_dict:
        outfile = Pickle_Helper(outPickleFile)
        outfile.dump(psd_data, "data")
        outfile.save()
    return len(psd_data)


# Creating ReUsable Object
rl = PSDRules()
input_preprocessor = PSDInputPreprocessor(opt, core_nlp_url)
rl.load(opt.build_folder+"dicts/psd_rule_f")
fragment_to_node_converter = PSDReCategorizor(from_file=False, path=opt.build_folder+"dicts/psd_recategorization", training=False)
# Creating ReUsable Object
# initializer = lasagne.init.Uniform()
non_rule_set_f = Pickle_Helper(opt.build_folder+"dicts/psd_non_rule_set")
non_rule_set = non_rule_set_f.load()["psd_non_rule_set"]
threshold = opt.threshold
# high_frequency, low_frequency = PSDRules.unmixe(non_rule_set, threshold)
high_frequency, low_frequency = PSDRules.unmixe(non_rule_set, threshold)
# expansion for a dictionary, add both constant and non-constant unaligned nodes
high_freq = {**high_frequency}
logger.info(
    "initial converted,threshold={},len(non_rule_set)={},high_frequency={},low_frequency={},high_freq={}".format(
        threshold, len(non_rule_set), len(high_frequency),
        len(low_frequency), len(high_freq)))

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
        word_dict = initial_dict(opt.build_folder+"dicts/word_dict", with_unk=True)
        char_dict = initial_dict(opt.build_folder+"dicts/char_dict", with_unk=True)
        lemma_dict = initial_dict(opt.build_folder+"dicts/lemma_dict", with_unk=True)
        source_pos_dict = initial_dict(opt.build_folder+"dicts/pos_dict", with_unk=True)
        ner_dict = initial_dict(opt.build_folder+"dicts/ner_dict", with_unk=True)  # from stanford

    # for target, pos, lem sense
    psd_high_dict = initial_dict(opt.build_folder+"dicts/psd_high_dict", with_unk=True)
    target_pos_dict = initial_dict(opt.build_folder+"dicts/psd_target_pos_dict", with_unk=True)

    rel_dict = initial_dict(opt.build_folder+"dicts/psd_rel_dict", with_unk=True)

    sense_dict = initial_dict(opt.build_folder+"dicts/psd_sense_dict", with_unk=True)
    # prepare psd_high_dict and psd_lemma_dict, make sure the the high_dict share the same part of high_freq id in the lemma_dict
    for uni in high_freq:
        # for high_dict part, we must keep the same ids
        le = uni.le
        id = lemma_dict.add(le)
        psd_high_dict.add(le, id)

    # PSD, there is no unalgined nodes, all lemma, or target ids can just filled nby the all the nodes in training data
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

    logger.info("len(word_dict)={},len(char_dict={}, len(lemma_dict)={}, len(source_pos_dict)={}, len(target_pos_dict)={}, len(ner_dict)={},len(rel_dict)={},len(sense_dict)={}, threshold={}".format(len(word_dict),len(char_dict),len(lemma_dict), len(source_pos_dict), len(target_pos_dict), len(ner_dict), len(rel_dict), len(sense_dict), threshold))
    logger.info("char_dict:\n {}".format(char_dict))
    logger.info("source_pos_dict:\n {}".format(source_pos_dict))
    logger.info("target_pos_dict:\n {}".format(target_pos_dict))
    logger.info("ner_dict:\n {}".format(ner_dict))
    logger.info("sense_dict:\n {}".format(sense_dict))
    logger.info("rel_dict:\n {}".format(rel_dict))

    # rel_dict = rel_dict.pruneByThreshold(threshold)
    rel_dict = rel_dict.prune(90)
    # for psd sense, we try to only use the top 20, threshold is 20 for it
    # we will not use prune for sense, we only sense before f10
    #sense_dict = sense_dict.pruneByThreshold(20)
    # prune by size
    sense_dict = sense_dict.prune(25)

    word_dict.save()
    char_dict.save()
    lemma_dict.save()
    source_pos_dict.save()
    target_pos_dict.save()
    ner_dict.save()
    rel_dict.save()
    sense_dict.save()
    psd_high_dict.save()

    logger.info("len(word_dict)={},len(char_dict={}, len(lemma_dict)={}, len(source_pos_dict)={}, len(target_pos_dict)={}, len(ner_dict)={},len(rel_dict)={},len(sense_dict)={}, threshold={}".format(len(word_dict),len(char_dict),len(lemma_dict), len(source_pos_dict), len(target_pos_dict), len(ner_dict), len(rel_dict), len(sense_dict), threshold))
    logger.info("source_pos_dict:\n {}".format(source_pos_dict))
    logger.info("target_pos_dict:\n {}".format(target_pos_dict))
    logger.info("ner_dict:\n {}".format(ner_dict))
    logger.info("sense_dict:\n {}".format(sense_dict))
    logger.info("rel_dict:\n {}".format(rel_dict))
    logger.info("psd_high_dict:\n {}".format(psd_high_dict))
else:

    if opt.merge_common_dicts:
        word_dict = Dict(opt.build_folder+"dicts/word_dict")
        lemma_dict = Dict(opt.build_folder+"dicts/lemma_dict")
        source_pos_dict = Dict(opt.build_folder+"dicts/pos_dict")
        ner_dict = Dict(opt.build_folder+"dicts/ner_dict")
        char_dict = Dict(opt.build_folder+"dicts/char_dict")
    else:
        word_dict = Dict(opt.build_folder+"dicts/word_dict")
        lemma_dict = Dict(opt.build_folder+"dicts/lemma_dict")
        source_pos_dict = Dict(opt.build_folder+"dicts/source_pos_dict")
        ner_dict = Dict(opt.build_folder+"dicts/ner_dict")
        char_dict = Dict(opt.build_folder+"dicts/char_dict")

    target_pos_dict = Dict(opt.build_folder+"dicts/psd_target_pos_dict")
    rel_dict = Dict(opt.build_folder+"dicts/psd_rel_dict")
    sense_dict = Dict(opt.build_folder+"dicts/psd_sense_dict")
    psd_high_dict = Dict(opt.build_folder+"dicts/psd_high_dict")

    word_dict.load()
    char_dict.load()
    lemma_dict.load()
    source_pos_dict.load()
    target_pos_dict.load()
    ner_dict.load()
    rel_dict.load()
    psd_high_dict.load()
    sense_dict.load()

fragment_to_node_converter = PSDReCategorizor(from_file=True, path=opt.build_folder+"dicts/psd_recategorization", training=False)

logger.info("len(word_dict)={},len(char_dict={}, len(lemma_dict)={}, len(source_pos_dict)={}, len(target_pos_dict)={}, len(ner_dict)={},len(rel_dict)={},len(sense_dict)={}, threshold={}".format(len(word_dict),len(char_dict),len(lemma_dict), len(source_pos_dict), len(target_pos_dict), len(ner_dict), len(rel_dict), len(sense_dict), threshold))

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
