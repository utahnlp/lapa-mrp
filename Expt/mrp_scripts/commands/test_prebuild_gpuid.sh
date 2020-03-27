#!/bin/bash

# source env.sh
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "$1" = /* ]]; then
  config_data=$1
else
  config_data=$DIR/$1
fi

### configurate data directory
if [ ! -f ${config_data} ]; then
  echo "${config_data} doesn't exist"
  exit $?
else
  . ${config_data}
  echo "run ${config_data}"
fi

### CHECK WORK & DATA DIR
if [ ! -e ${EXP_DIR} ]; then
  echo "original model not exist"
  exit -1
fi

# the restore_model_name
model_name=$2
gpuid=$3

### How to restore the model and continue to train
### TRAIN for task1
pargs="
--char_dim=${CHAR_DIM} \
--char_encoder_config=${CHAR_ENCODER_CONFIG} \
--dm_cat_bias=${DM_CAT_BIAS} \
--psd_sense_bias=${PSD_SENSE_BIAS} \
--psd_target_pos_bias=${PSD_TARGET_POS_BIAS} \
--dm_sense_bias=${DM_SENSE_BIAS} \
--dm_target_pos_bias=${DM_TARGET_POS_BIAS} \
--frames=${FRAMES} \
--normalize_mod=${NORMALIZE_MOD} \
--posterior_amr_encoder=${POSTERIOR_AMR_ENCODER} \
--optim_json_configs=${OPTIM_JSON_CONFIGS} \
--use_src_encs_for_root=${USE_SRC_ENCS_FOR_ROOT} \
--use_src_encs_for_rel=${USE_SRC_ENCS_FOR_REL} \
--use_src_encs_for_posterior=${USE_SRC_ENCS_FOR_POSTERIOR} \
--root_snt_encoder=${ROOT_SNT_ENCODER} \
--rel_snt_encoder=${REL_SNT_ENCODER} \
--posterior_snt_encoder=${POSTERIOR_SNT_ENCODER} \
--concept_snt_encoder=${CONCEPT_SNT_ENCODER} \
--summary_dir=${SUMMARY_DIR} \
--debug_size=${DEBUG_SIZE} \
--optim_scheduler_name=${OPTIM_SCHEDULER_NAME} \
--gradient_accumulation=${GRADIENT_ACCUMULATION} \
--warmup_proportion=${WARMUP_PROPORTION} \
--max_bert_seq_length=${MAX_BERT_SEQ_LENGTH} \
--bert_model=${BERT_MODEL} \
--mask_pre_unaligned=${MASK_PRE_UNALIGNED} \
--result_folder=${RESULT_FOLDER} \
--build_folder=${BUILD_FOLDER} \
--suffix=${SUFFIX} \
--folder=${FOLDER} \
--jamr=${JAMR} \
--save_to=${SAVE_TO} \
--restore_from=${EXP_MODELS}/$2 \
--get_wiki=${GET_WIKI} \
--get_sense=${GET_SENSE} \
--cat_bias=${CAT_BIAS} \
--lemma_bias=${LEMMA_BIAS} \
--independent_posterior=${INDEPENDENT_POSTERIOR} \
--train_posterior=${TRAIN_POSTERIOR} \
--alpha_dropout=${ALPHA_DROPOUT} \
--initialize_word=${INITIALIZE_WORD} \
--rel_enlayers=${REL_ENLAYERS} \
--root_enlayers=${ROOT_ENLAYERS} \
--txt_enlayers=${TXT_ENLAYERS} \
--amr_enlayers=${AMR_ENLAYERS} \
--txt_rnn_size=${TXT_RNN_SIZE} \
--rel_rnn_size=${REL_RNN_SIZE} \
--amr_rnn_size=${AMR_RNN_SIZE} \
--rel=${REL} \
--word_dim=${WORD_DIM} \
--dim=${DIM} \
--pos_dim=${POS_DIM} \
--ner_dim=${NER_DIM} \
--cat_dim=${CAT_DIM} \
--rel_dim=${REL_DIM} \
--brnn=${BRNN} \
--weight_decay=${WEIGHT_DECAY} \
--retrain_all=${RETRAIN_ALL} \
--batch_size=${BATCH_SIZE} \
--epochs=${EPOCHS} \
--start_epoch=${START_EPOCH} \
--optim=${OPTIM} \
--learning_rate=${LEARNING_RATE} \
--max_grad_norm=${MAX_GRAD_NORM} \
--dropout=${DROPOUT} \
--gumbel=${GUMBEL} \
--sink=${SINKHORN} \
--sink_t=${SINK_T} \
--prior_t=${PRIOR_T} \
--sink_re=${SINK_RE} \
--learning_rate_decay=${LEARNING_RATE_DECAY} \
--start_decay_at=${START_DECAY_AT} \
--emb_independent=${EMB_INDEPENDENT} \
--gpus=${GPUS} \
--from_gpus=${FROM_GPUS} \
--log_per_epoch=${LOG_PER_EPOCH} \
--renyi_alpha=${RENYI_ALPHA}
"

echo $pargs
pushd $ROOT_DIR
CUDA_VISIBLE_DEVICES=$gpuid python src/generate.py $pargs &> ${EXP_DIR}/test_pretrained_$2.log
popd
