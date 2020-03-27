# the root dir of this code rep¬
ROOT_DIR=$CODE_BASE/amr/

# Options for Experiments
EXP_ROOT_DIR=$CODE_BASE/amr/Expt/workdir/
# https://stackoverflow.com/questions/965053/extract-filename-and-extension-in-bash
config_name=`basename "$1"`
EXP_NAME="${config_name%.*}"
echo $EXP_NAME
EXP_DIR=$EXP_ROOT_DIR/base_bert/$EXP_NAME/
EXP_MODELS=$EXP_DIR/models/
EXP_SUMMARY=$EXP_DIR/summary/
EXP_RESULTS=$EXP_DIR/results/

# Options for data file paths
# .txt_pre_processed, suffix of files to combine, default ".txt_pre_processed"
SUFFIX=.txt_pre_processed
# the folder for storing AMRs, subdirs are /train/ /dev/ /test/
FOLDER=$CODE_BASE/amr_data/e25/data/alignments/split/
# the folder for generated AMRs
RESULT_FOLDER=$EXP_RESULTS
# the folder for building data dnd rules/
BUILD_FOLDER=$CODE_BASE/amr/build/amr2eng_bert/
# whether to use fixed alignment, default False
JAMR=
# the folder for saving model
SAVE_TO=$EXP_MODELS
# If training from a checkpoint, then load from
RESTORE_FROM=


# Model options
# whether use wiki
GET_WIKI=x
# whether use sense
GET_SENSE=x
# Whether bias category, default = 1
CAT_BIAS=1
# Whether lemma bias
LEMMA_BIAS=0
# Whether use different snt encoder for independent posterior
INDEPENDENT_POSTERIOR=1
# Whether to keep train_posterior, default = True
TRAIN_POSTERIOR=1
# alpha dropout, used for data dropout, default = True
ALPHA_DROPOUT=0.1
# whether initialize word
INITIALIZE_WORD=1

# options for layers
# Number of layers in the rel LSTM encoder/decoder, int, 2
REL_ENLAYERS=2
# Number of layers in the root LSTM encoder/decoder
ROOT_ENLAYERS=1
# Number of layers in the concept LSTM encoder/decoder
TXT_ENLAYERS=1
# Number of layers in the amr LSTM encoder/decoder
AMR_ENLAYERS=1
# The rnn hidden size of txt_rnn, default is 512
TXT_RNN_SIZE=512
# The rnn hidden size of rel_rnn, default is 512
REL_RNN_SIZE=512
# The rnn hidden size of amr_rnn, default is 200
AMR_RNN_SIZE=200

# Whether to train relation /root identification, float, also the weight for rel loss
REL=1.0
# word_embedding size, default 300
WORD_DIM=300
# lemma_dim or high dim, default 200
DIM=200
# Pos embedding size, default 32
POS_DIM=32
# Ner embedding size, default 16
NER_DIM=16
# Category embedding size, default 32
CAT_DIM=32
# mixed amr node and text representation dimension, default 200
REL_DIM=16
# whether use bidiretional encoder, default True
BRNN=x

# Optimization and training options
# L2 weight decay, default 0.00001
WEIGHT_DECAY=0.0
# Whether to train all paramaters,  usefull when reloading model for train
RETRAIN_ALL=
# Maximum batch size, default 64
BATCH_SIZE=64
# epochs to train, default 30
EPOCHS=30
# The epoch from which to start, default = 1
START_EPOCH=1
# Optimization Methods, default is adam, support sgd,adagrad, adadelta, adam
OPTIM=BertAdam
# learning rate. If adagrad/adadelta/adam is used, then this is the global learning rate. Recommended settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1"
LEARNING_RATE=0.01
# If the norm of the gradient vector exceeds this, renormalize it to have the norm equal to max_grad_norm
MAX_GRAD_NORM=10
# Dropout probability; applied between LSTM stacks and some MLPs. default, 0.2
DROPOUT=0.2
# whether using gumbel softmax, default is Ture
GUMBEL=x
# gumbel-sinkhorn procedure, Number of iterations, default is 10
SINKHORN=10
# gumbel sinkhorn temperature, default = 1
SINK_T=1
# Prior tempeture for gumbel-sinkhorn, default = 5
PRIOR_T=1
# gumbel-sinkhorn finite step regularizor  penalizing non-double-stochasitcit
SINK_RE=10
# Decay learning rate by this much if (i) perplexity does not drecrease on the validation set or (ii) epoch has gone past the start_decay_at_limit, default = 0.98
LEARNING_RATE_DECAY=0.98
# start decay after this epoch, default= 5
START_DECAY_AT=5
# Whether relation system use independent embeddings, default=1
EMB_INDEPENDENT=1

# GPUS
# gpu ids to use, default = -2, means loaded by outside gpu scheudler
GPUS=-2
# from_gpus, gpuid will be used when saving model, if gpu id is different from saving time, we need to specify this
FROM_GPUS=
# log per epoch, print stats at this interval, default 10
LOG_PER_EPOCH=10
# renyi_alpha, default=0.5, parameter of renyi_alpha relaxation, which is alternative to hierachical relaxation in the paper, used for soft version loss
RENYI_ALPHA=0.5
# weights for masking pre_unaligned
MASK_PRE_UNALIGNED=1e6
# the name for pretrained bert model, default empty
BERT_MODEL=bert-base-cased
# max_bert_seq_length to use for bert seq, include special bert tokens, default = 64
MAX_BERT_SEQ_LENGTH=64
# warmup_proportion used for linear warmup
WARMUP_PROPORTION=0.1
# gradient_accumulation for N steps, then do
GRADIENT_ACCUMULATION=2
# optim scheduler name to use for learning rate scheduling, default = none, warmup_cosine, warmup_constent, warmup_linear
OPTIM_SCHEDULER_NAME=warmup_linear
# debug_size, defalue = -1, means no debug, positive value means in debug mode
DEBUG_SIZE=-1
# summary dir to write tensor board summary
SUMMARY_DIR=${EXP_SUMMARY}
# concept_snt_encoder, Model:Share:ARG
CONCEPT_SNT_ENCODER=bert-base-cased:concept:d:1
# posterior_snt_encoder, Model:Share:ARG
POSTERIOR_SNT_ENCODER=bert-base-cased:posterior:d:1
# rel_snt_encoder, Model:Share:ARG
REL_SNT_ENCODER=rnn
# root_snt_encoder, Model:Share:ARG
ROOT_SNT_ENCODER=rnn
# use_src_encs_for_posterior
USE_SRC_ENCS_FOR_POSTERIOR=
# use_src_encs_for_rel
USE_SRC_ENCS_FOR_REL=
# use_src_encs_for_root
USE_SRC_ENCS_FOR_ROOT=
# optim json configs
OPTIM_JSON_CONFIGS='{"[bert_model|_scalar_mix]":{"lr":0.0001}}'
# posterior_amr_encoder for amr
POSTERIOR_AMR_ENCODER=rnn
# whether normalize mod
NORMALIZE_MOD=x
# meaning representation frames to parse, seperate by comma, default i s'amr'
FRAMES=amr
# whether bias dm target pos
DM_TARGET_POS_BIAS=1
# whether bias dm sense
DM_SENSE_BIAS=1
# whether bias psd target pos bias
PSD_TARGET_POS_BIAS=1
# whether bias psd sense bias
PSD_SENSE_BIAS=1
# whether bias dm cat bias
DM_CAT_BIAS=1
# char dim
CHAR_DIM=64
# char encoder config
CHAR_ENCODER_CONFIG=
