import torch
from pytorch_transformers.modeling_bert import BertModel
from overrides import overrides
import torch.nn as nn
from parser.modules.helper_module import *
from pytorch_transformers.modeling_bert import BertModel
from allennlp.modules.seq2seq_encoders.stacked_self_attention import *
from torch.nn.utils.rnn import PackedSequence
from utility.constants import *
import numpy as np
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.tokenization_xlnet import XLNetTokenizer
import logging

logger = logging.getLogger("mrp.bert_utils")

class MRPBertTokenizer(BertTokenizer):

    def __init__(self, *args, **kwargs):
        super(MRPBertTokenizer, self).__init__(*args, **kwargs)

    @overrides
    def tokenize(self, tokens, split=True, max_tokenized_length = 64):
        # https://github.com/huggingface/pytorch-transformers/blob/f31154cb9df44b9535bd21eb5962e7a91711e9d1/examples/utils_glue.py
        tokens = ['[CLS]'] + tokens
        if not split:
            split_tokens = [t if t in self.vocab else '[UNK]' for t in tokens]
            gather_indexes = None
        else:
            split_tokens, _gather_indexes = [], []
            for token in tokens:
                indexes = []
                for i, sub_token in enumerate(self.wordpiece_tokenizer.tokenize(token)):
                    if len(split_tokens) < max_tokenized_length-1:
                        indexes.append(len(split_tokens))
                        split_tokens.append(sub_token)
                    else:
                        # cut it
                        pass
                _gather_indexes.append(indexes)
            # last token is always [SEP], we add it to split_token, not add it to gather index, whichwill be discarded
            split_tokens.append('[SEP]')
            _gather_indexes = _gather_indexes[1:]
            max_index_list_len = max(len(indexes) for indexes in _gather_indexes)
            gather_indexes = np.zeros((len(_gather_indexes), max_index_list_len))
            for i, indexes in enumerate(_gather_indexes):
                for j, index in enumerate(indexes):
                    gather_indexes[i, j] = index

        token_ids = np.array(self.convert_tokens_to_ids(split_tokens))
        return token_ids, gather_indexes


class MRPXLNetTokenizer(XLNetTokenizer):

    def __init__(self, *args, **kwargs):
        super(MRPXLNetTokenizer, self).__init__(*args, **kwargs)

    @overrides
    def tokenize(self, tokens, split=True, max_tokenized_length = 64):
        if not split:
            split_tokens = [t if t in self.vocab else '[UNK]' for t in tokens]
            gather_indexes = None
        else:
            split_tokens, _gather_indexes = [], []
            for token in tokens:
                indexes = []
                for i, sub_token in enumerate(self._tokenize(token)):
                    if len(split_tokens) < max_tokenized_length-2:
                        indexes.append(len(split_tokens))
                        split_tokens.append(sub_token)
                    else:
                        # cut it
                        pass
                _gather_indexes.append(indexes)
            # last token is always [CLS] for xlnet, we add it to split_token, not add it to gather index, whichwill be discarded
            split_tokens.append('[SEP]')
            split_tokens.append('[CLS]')
            _gather_indexes = _gather_indexes[1:]
            max_index_list_len = max(len(indexes) for indexes in _gather_indexes)
            gather_indexes = np.zeros((len(_gather_indexes), max_index_list_len))
            for i, indexes in enumerate(_gather_indexes):
                for j, index in enumerate(indexes):
                    gather_indexes[i, j] = index

        token_ids = np.array(self.convert_tokens_to_ids(split_tokens))


class BertEncoderUtils(BertModel):

    #def build_bert_cache(snt_dict_file, max_snt_length, output_cache_file):
    #    """
    #    Go through all the snts in the dataset, save into the cache, every snt it has a exampleid
    #    """
    #    self.logger.info('Prepare bert embeddings for {} with ELMo_Utils ...'.format(snt_dict_file))
    #    with open(snt_dict_file, 'r') as fin, h5py.File(output_cache_file, 'w') as fout:
    #        lm_embeddings_h5 = fout.create_group('lm_embeddings')
    #        lengths_h5 = fout.create_group('lengths')
    #        mask_h5 = fout.create_group('mask')
    #        batch_snts = []
    #        start_snt_id_in_batch = 0
    #        SNT_BATCH_SIZE = 10
    #        for line in tqdm(fin, total=get_num_lines(snt_dict_file)):
    #            sentence = line.strip().split()
    #            batch_snts.append(sentence)
    #            length = len(batch_snts)
    #            if length >= SNT_BATCH_SIZE:
    #                start_snt_id_in_batch += self.consume_batch_snts(sess, ids_placeholder, ops, batch_snts,max_snt_length, start_snt_id_in_batch, lm_embeddings_h5, lengths_h5, mask_h5)
    #                batch_snts = []
    #        if len(batch_snts) > 0:
    #            start_snt_id_in_batch += self.consume_batch_snts(sess, ids_placeholder, ops, batch_snts,max_snt_length, start_snt_id_in_batch, lm_embeddings_h5, lengths_h5, mask_h5)
    #            batch_snts = []
    #       logger.info("Finished Bert embeddings for {} senencesm in {}".format(start_snt_id_in_batch, output_cache_file))

    @staticmethod
    def encodeSequence(bert_model, packed_input: PackedSequence, bertBatch = None, bertIndexBatch=None, _scalar_mix = None, bert_layers_mix_indices = [], sub_word_handler = 'avg_pool'):
        # pack[src_len*batch, n_feature]
        # token_sub_word_index, [src_len x bathe_size x max_subwords]
        # becaus ethe bert use batch_first
        token_sub_word_index, lengths = unpack(bertIndexBatch, batch_first=True)
        # bertBatch: batch_size x max_bert_sequence
        # input masks
        input_masks = (bertBatch != BERT_PAD_INDEX).long()
        # for single sentence, token_type_id are all zeors,
        # for two sentences, second sentence will be ones
        # all_encode_layers :  layers x batch_data x max_bert_sequence x hidden
        # sequence_output, pooled_output, (hidden_states), (attentions)
        last_encoder_layer, pooled_output, all_encoder_layers,_ = bert_model(bertBatch, token_type_ids=None, attention_mask=input_masks)
        # only select the index for origin tokens, borrow from allennlp implments
        # https://github.com/allenai/allennlp/blob/e9287d4d48d2d980226a4acf70d03eedda67c548/allennlp/modules/token_embedders/bert_token_embedder.py
        if _scalar_mix is not None:
            # when not do layer norm, input_mask is not used
            # all_encoder_layers is a list
            selected_layers = [all_encoder_layers[i] for i in bert_layers_mix_indices]
            mix = _scalar_mix(selected_layers, input_masks)
        else:
            # use the first selected bert_layer_mix_indices
            if len(bert_layers_mix_indices)> 0:
                mix = all_encoder_layers[bert_layers_mix_indices[0]]
            else:
                mix = last_encoder_layer

        #logger.info("batch_input:{} , bertBatch:{}, offsets:{}, input_masks:{}, all_encode_layers_size:{} x {},  mix_size:{}".format(batch_input, bertBatch, offsets, input_masks, len(all_encoder_layers), all_encoder_layers[0].size(), mix.size()))
         # At this point, mix is (max_bert_length, batch_size * d1 * ... * dn, embedding_dim)
        if token_sub_word_index is None or sub_word_handler == None:
            # bertBatch: batch, max_bert_sequence
            # https://allenai.github.io/allennlp-docs/api/allennlp.nn.util.html
            # bertEncodes: max_bert_sequence x batch, hidden
            # when no offsets, we assump it is 1-to-1 mapping
            bert_encodes = util.uncombine_initial_dims(mix, bertBatch.size()).transpose(0, 1).view(-1, mix.size(-1))
            # here if no offfset, bert_encodes is padded encoding, wrong here
            return pack(bert_encodes, lengths), pooled_output
        else:
            if sub_word_handler == "first_token":
                offsets = token_sub_word_index[:,0].squeeze(1)
                # offsets: src_len_batch
                with torch.no_grad():
                    # https://allenai.github.io/allennlp-docs/api/allennlp.nn.util.html
                    packed_offset = PackedSequence(offsets, packed_input.batch_sizes)
                    # padded_offset : [origin_max_src_length, batch_size, dim]
                    padded_offset = unpack(packed_offset)[0].unsqueeze(2).expand(-1,-1,mix.size(-1))
                #range_vector = util.get_range_vector(padded_offset.size(0),
                #                                     device=util.get_device_of(mix)).unsqueeze(1)
                # selected embeddings is also (origin_max_src_lengh, batch_size * d1 * ... * dn, hidden)
                # https://pytorch.org/docs/stable/torch.html#torch.gather
                # logger.info("packed_offset:{} ,padded_offset:{}, mix={}".format(packed_offset, padded_offset, mix))
                # padded_offset : [origin_max_src_length, batch_size, dim]
                # mix :[batch_data x max_bert_squence_len x hidden]
                selected_embeddings = mix.gather(0, padded_offset)
                # padded not batch_first, [max_src_len, batch, hidden]
                return pack(selected_embeddings, lengths), pooled_output
            elif sub_word_handler == "avg_pool":
                # in stog, it only use the last layer of bert
                avg_pool_token_encodings = BertEncoderUtils.average_pooling(mix, token_sub_word_index)
                return pack(avg_pool_token_encodings, lengths, batch_first=True), pooled_output
            elif sub_word_handler == "max_pool":
                max_pool_token_encodings = BertEncoderUtils.max_pooling(mix, token_sub_word_index)
                return pack(max_pool_token_encodings, lengths, batch_first=True), pooled_output
            else:
                raise NotImplementedError("not implemented subword handler = {}".format(sub_word_handler))


    def average_pooling(encoded_layers, token_subword_index):
        batch_size, num_tokens, num_subwords = token_subword_index.size()
        batch_index = torch.arange(batch_size).view(-1, 1, 1).type_as(token_subword_index)
        token_index = torch.arange(num_tokens).view(1, -1, 1).type_as(token_subword_index)
        _, num_total_subwords, hidden_size = encoded_layers.size()
        expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
            batch_size, num_tokens, num_total_subwords, hidden_size)
        # [batch_size, num_tokens, num_subwords, hidden_size]
        token_reprs = expanded_encoded_layers[batch_index, token_index, token_subword_index]
        subword_pad_mask = token_subword_index.eq(0).unsqueeze(3).expand(
            batch_size, num_tokens, num_subwords, hidden_size)
        token_reprs.masked_fill_(subword_pad_mask, 0)
        # [batch_size, num_tokens, hidden_size]
        sum_token_reprs = torch.sum(token_reprs, dim=2)
        # [batch_size, num_tokens]
        num_valid_subwords = token_subword_index.ne(0).sum(dim=2)
        pad_mask = num_valid_subwords.eq(0).long()
        # Add ones to arrays where there is no valid subword.
        divisor = (num_valid_subwords + pad_mask).unsqueeze(2).type_as(sum_token_reprs)
        # [batch_size, num_tokens, hidden_size]
        avg_token_reprs = sum_token_reprs / divisor
        return avg_token_reprs

    def max_pooling(encoded_layers, token_subword_index):
        batch_size, num_tokens, num_subwords = token_subword_index.size()
        batch_index = torch.arange(batch_size).view(-1, 1, 1).type_as(token_subword_index)
        token_index = torch.arange(num_tokens).view(1, -1, 1).type_as(token_subword_index)
        _, num_total_subwords, hidden_size = encoded_layers.size()
        expanded_encoded_layers = encoded_layers.unsqueeze(1).expand(
            batch_size, num_tokens, num_total_subwords, hidden_size)
        # [batch_size, num_tokens, num_subwords, hidden_size]
        token_reprs = expanded_encoded_layers[batch_index, token_index, token_subword_index]
        subword_pad_mask = token_subword_index.eq(0).unsqueeze(3).expand(
            batch_size, num_tokens, num_subwords, hidden_size)
        token_reprs.masked_fill_(subword_pad_mask, -float('inf'))
        # [batch_size, num_tokens, hidden_size]
        max_token_reprs, _ = torch.max(token_reprs, dim=2)
        # [batch_size, num_tokens]
        num_valid_subwords = token_subword_index.ne(0).sum(dim=2)
        pad_mask = num_valid_subwords.eq(0).unsqueeze(2).expand(
            batch_size, num_tokens, hidden_size)
        max_token_reprs.masked_fill(pad_mask, 0)
        return max_token_reprs
