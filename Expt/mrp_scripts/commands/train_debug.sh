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
if [ -e ${EXP_DIR} ]; then
  today=`date +%m-%d.%H:%M`
  mv ${EXP_DIR} ${EXP_DIR%?}_${today}
  echo "rename original training folder to "${EXP_DIR%?}_${today}
fi

mkdir -p ${EXP_DIR}
mkdir -p ${EXP_MODELS}
mkdir -p ${EXP_SUMMARY}
mkdir -p ${EXP_RESULTS}

### How to restore the model and continue to train
### TRAIN for task1
pargs="--train \
--debug_size=${DEBUG_SIZE} \
--optim_scheduler_name=${OPTIM_SCHEDULER_NAME} \
--gradient_accumulation=${GRADIENT_ACCUMULATION} \
--warmup_proportion=${WARMUP_PROPORTION} \
--bert_num_top_layers_mix=${BERT_NUM_TOP_LAYERS_MIX} \
--summary_dir=${SUMMARY_DIR} \
--max_bert_seq_length=${MAX_BERT_SEQ_LENGTH} \
--bert_model=${BERT_MODEL} \
--mask_pre_unaligned=${MASK_PRE_UNALIGNED} \
--build_folder=${BUILD_FOLDER} \
--suffix=${SUFFIX} \
--folder=${FOLDER} \
--jamr=${JAMR} \
--save_to=${SAVE_TO} \
--restore_from=${RESTORE_FROM} \
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
CUDA_LAUNCH_BLOCKING=1 python src/train.py $pargs 2>&1 &> ${EXP_DIR}/train.log
popd

