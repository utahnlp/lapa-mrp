from utility.amr_utils.amr import *
from utility.data_helper import *
import torch
import logging

logger = logging.getLogger("mrp.Dict")

def seq_to_id(dictionary,seq):
    id_seq = []
    freq_seq = []
    for i in seq:
        id_seq.append(dictionary[i])
        freq_seq.append(dictionary.frequencies[dictionary[i]])
    return id_seq,freq_seq

def read_dicts(folder_name="", frame="amr"):
    if frame == "amr":
        dicts = read_amr_dicts(folder_name)
    elif frame == "dm":
        dicts = read_dm_dicts(folder_name)
    elif frame == "eds":
        dicts = read_eds_dicts(folder_name)
    elif frame == "psd":
        dicts = read_psd_dicts(folder_name)
    else:
        raise NotImplementedError("{} is not supported".format(frame))

    return dicts

def read_all_dicts(folder_name="",frames=["amr"]):
    dicts = read_common_dicts(folder_name)
    for frame in frames:
        dicts.update(read_dicts(folder_name,frame))
    return dicts

def read_common_dicts(folder_name=""):
    logger.info("Begin loading dicts from {}".format(folder_name))
    word_dict = Dict(folder_name+"dicts/word_dict")
    char_dict = Dict(folder_name+"dicts/char_dict")
    lemma_dict = Dict(folder_name+"dicts/lemma_dict")
    ner_dict = Dict(folder_name+"dicts/ner_dict")
    pos_dict = Dict(folder_name+"dicts/pos_dict")
    word_dict.load()
    char_dict.load()
    lemma_dict.load()
    ner_dict.load()
    pos_dict.load()
    dicts = dict()
    dicts["word_dict"]= word_dict
    dicts["char_dict"]= char_dict
    dicts["lemma_dict"]= lemma_dict
    dicts["ner_dict"]= ner_dict
    dicts["pos_dict"]= pos_dict
    logger.info("word_dict={}\t char_dict={}\t lemma_dict = {} \t ner_dict={} \t pos_dict={}".format(len(word_dict), len(char_dict), len(lemma_dict), len(ner_dict), len(pos_dict)))
    return dicts


def read_psd_dicts(folder_name=""):

    logger.info("Begin loading dicts from {}".format(folder_name))
    psd_target_pos_dict = Dict(folder_name+"dicts/psd_target_pos_dict")
    psd_rel_dict = Dict(folder_name+"dicts/psd_rel_dict")
    psd_sense_dict = Dict(folder_name+"dicts/psd_sense_dict")
    psd_high_dict = Dict(folder_name+"dicts/psd_high_dict")

    psd_target_pos_dict.load()
    psd_rel_dict.load()
    psd_sense_dict.load()
    psd_high_dict.load()
    dicts = dict()

    dicts["psd_rel_dict"] = psd_rel_dict
    dicts["psd_target_pos_dict"] = psd_target_pos_dict
    dicts["psd_sense_dict"] = psd_sense_dict
    dicts["psd_high_dict"] = psd_high_dict
    logger.info("psd_target_pos_dict={}\t psd_sense_dict={}\t psd_high_dict={}, psd_rel_dict={}".format(
        len(psd_target_pos_dict), len(psd_sense_dict),len(psd_high_dict),len(psd_rel_dict)))
    return dicts


def read_dm_dicts(folder_name=""):

    logger.info("Begin loading dicts from {}".format(folder_name))
    dm_cat_dict = Dict(folder_name+"dicts/dm_cat_dict")
    dm_high_dict = Dict(folder_name+"dicts/dm_high_dict")
    #dm_high_le_dict = Dict(folder_name+"dicts/dm_high_le_dict")
    dm_target_pos_dict = Dict(folder_name+"dicts/dm_target_pos_dict")
    dm_rel_dict = Dict(folder_name+"dicts/dm_rel_dict")
    dm_sense_dict = Dict(folder_name+"dicts/dm_sense_dict")

    dm_target_pos_dict.load()
    dm_rel_dict.load()
    dm_cat_dict.load()
    dm_sense_dict.load()
    dm_high_dict.load()
    # dm_high_le_dict.load()
    dicts = dict()

    dicts["dm_rel_dict"] = dm_rel_dict
    dicts["dm_cat_dict"] = dm_cat_dict
    dicts["dm_target_pos_dict"] = dm_target_pos_dict
    dicts["dm_sense_dict"] = dm_sense_dict
    dicts["dm_high_dict"] = dm_high_dict
    # dicts["dm_high_le_dict"] = dm_high_le_dict
    logger.info("dm_target_pos_dict={}\t dm_cat_dict={}\t dm_high_dict = {}\t dm_high_le_dict = {}\t dm_sense_dict={}\t dm_rel_dict={}".format(
        len(dm_target_pos_dict), len(dm_cat_dict), len(dm_high_dict), 0, len(dm_sense_dict), len(dm_rel_dict)))
    return dicts


def read_eds_dicts(folder_name=""):

    logger.info("Begin loading dicts from {}".format(folder_name))
    eds_cat_dict = Dict(folder_name+"dicts/eds_cat_dict")
    eds_high_dict = Dict(folder_name+"dicts/eds_high_dict")
    eds_rel_dict = Dict(folder_name+"dicts/eds_rel_dict")
    eds_sense_dict = Dict(folder_name+"dicts/eds_sense_dict")

    eds_target_pos_dict.load()
    eds_rel_dict.load()
    eds_cat_dict.load()
    eds_sense_dict.load()
    eds_high_dict.load()
    dicts = dict()

    dicts["eds_rel_dict"] = eds_rel_dict
    dicts["eds_cat_dict"] = eds_cat_dict
    dicts["eds_high_dict"] = eds_high_dict
    logger.info("eds_cat_dict={}\t eds_high_dict = {}\t eds_rel_dict={}".format(
        len(eds_cat_dict), len(eds_high_dict),len(dm_rel_dict)))
    return dicts

def read_amr_dicts(folder_name=""):

    logger.info("Begin loading dicts from {}".format(folder_name))
    amr_aux_dict = Dict(folder_name+"dicts/amr_aux_dict")
    amr_high_dict = Dict(folder_name+"dicts/amr_high_dict")
    amr_rel_dict = Dict(folder_name+"dicts/amr_rel_dict")
    amr_category_dict = Dict(folder_name+"dicts/amr_category_dict")

    amr_rel_dict.load()
    amr_category_dict.load()
    amr_high_dict.load()
    amr_aux_dict.load()
    dicts = dict()

    dicts["amr_rel_dict"] = amr_rel_dict
    dicts["amr_category_dict"] = amr_category_dict
    dicts["amr_aux_dict"] = amr_aux_dict
    dicts["amr_high_dict"] = amr_high_dict
    logger.info("amr_high_dict={}\t amr_category_dict={}\t amr_aux_dict={}\t amr_rel_dict={}".format(
        len(amr_high_dict), len(amr_category_dict), len(amr_aux_dict), len(amr_rel_dict)))
    return dicts

class Dict(object):
    def __init__(self, fileName,dictionary=None):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}

        # Special entries will not be pruned.
        self.special = []

        if dictionary :
            for label in dictionary:
                self.labelToIdx[label] = dictionary[label][0]
                self.idxToLabel[dictionary[label][0]] = label
                self.frequencies[dictionary[label][0]] = dictionary[label][1]
        self.fileName = fileName

    def size(self):
        return len(self.idxToLabel)
    
    def __len__(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def load(self, filename =None):
        if filename:
            self.fileName = filename
        else:
            filename = self.fileName 
        f = Pickle_Helper(filename) 
        data = f.load()
        self.idxToLabel=data["idxToLabel"]
        self.labelToIdx=data["labelToIdx"]
        self.frequencies=data["frequencies"]

    # Write entries to a file.
    def save(self, filename  =None):
        if filename:
            self.fileName = filename
        else:
            filename = self.fileName 
        f = Pickle_Helper(filename) 
        f.dump( self.idxToLabel,"idxToLabel")
        f.dump( self.labelToIdx,"labelToIdx")
        f.dump( self.frequencies,"frequencies")
        f.save()

    def lookup(self, key, default=None):
        try:
            return self.labelToIdx[key]
        except KeyError:
            if default: return default

            return self.labelToIdx[UNK_WORD]
    def __str__(self):
        out_str = []
        for k in self.frequencies:
            if k not in self.special:
                out_str.append(str(k) +" ,"+ self.idxToLabel[k]+": "+str(self.frequencies[k]))
        return " \n".join(out_str)
    def __getitem__(self, label,default=None):
        try:
            return self.labelToIdx[label]
        except KeyError:
            if default: return default

            return self.labelToIdx[UNK_WORD]
    
    def getLabel(self, idx, default=UNK_WORD):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default
        
    def __iter__(self): return self.labelToIdx.__iter__()
    def __next__(self): return self.labelToIdx.__next__()
    # Mark this `label` and `idx` as special (i.e. will not be pruned).
    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    # Mark all labels in `labels` as specials (i.e. will not be pruned).
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label, idx=None):
        return self.addWithFreq(label,1,idx)

    def addWithFreq(self, label, freq, idx=None):
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = freq
        else:
            self.frequencies[idx] += freq

        return idx

    def __setitem__(self, label, idx):
        self.add(label,idx)

    # Return a new dictionary with the `size` most frequent entries.
    def prune(self, size):
        if size >= self.size():
            return self

        # Only keep the `size` most frequent entries.
        freq = torch.Tensor(
                [self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        newDict = Dict(self.fileName)

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:size]:
            newDict.addWithFreq(self.idxToLabel[i.item()], freq = self.frequencies[i.item()])

        return newDict
    # Return a new dictionary with the `size` most frequent entries.
    def pruneByThreshold(self, threshold):
        # Only keep the `size` most frequent entries.
        high_freq = [ (self.frequencies[i],i) for i in range(len(self.frequencies)) if self.frequencies[i]>threshold]

        newDict = Dict(self.fileName)

        # Add special entries in all cases.
        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for freq,i in high_freq:
            newDict.add(self.idxToLabel[i])
            newDict.frequencies[newDict.labelToIdx[self.idxToLabel[i]]] = freq

        return newDict
    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord = UNK_WORD, bosWord=BOS_WORD, eosWord=EOS_WORD):
        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)
        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return torch.LongTensor(vec)

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop=[]):
        labels = []

        for i in idx:
            if i in stop:
                break
            labels += [self.getLabel(i)]

        return labels
