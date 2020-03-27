import torch
from overrides import overrides
import torch.nn as nn
from parser.modules.helper_module import *
from pytorch_transformers.modeling_bert import BertModel
from allennlp.modules.seq2seq_encoders.stacked_self_attention import *
from allennlp.data.tokenizers import CharacterTokenizer
from torch.nn.utils.rnn import PackedSequence
from utility.constants import *
import numpy as np
import logging

logger = logging.getLogger("mrp.char_utils")

class CharEncoderUtils:
    @staticmethod
    def encodeSequence(char_model, src_charBatch, char_emb):
        # pack[src_len*batch, n_feature]
        # src_len * batch, n_chaeacters, char_dim
        char_input = char_emb(src_charBatch.data)
        # input masks, (src_len x batch_size, nchar)
        input_masks = (src_charBatch.data!= PAD).long()
        token_encodings = char_model(char_input, input_masks)
        # src_len x batch x projection_dim
        return PackedSequence(token_encodings, src_charBatch.batch_sizes)

character_tokenizer = CharacterTokenizer(start_tokens=[BOS_WORD], end_tokens=[EOS_WORD])

class CharTokenizerUtils:

    @staticmethod
    def tokenize(snt_token, char_dict):
        char_array = []
        max_len = 0
        for t in snt_token:
            chars = character_tokenizer.tokenize(t)
            char_ids = [char_dict.add(c.text) for c in chars]
            char_array.append(char_ids)
            if max_len < len(char_ids):
                max_len = len(char_ids)

        chars_id_matrix = np.zeros((len(char_array), max_len))
        for i, ids in enumerate(char_array):
            for j, id in enumerate(ids):
                chars_id_matrix[i, j] = id
        return chars_id_matrix
