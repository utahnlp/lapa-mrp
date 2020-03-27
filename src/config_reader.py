import subprocess
import argparse
from utility.constants import *
from parser.Optim import *

#may add --evaluate
pargs='''
--train \
--normalize_mod=${NORMALIZE_MOD} \
--posterior_snt_encoder=${POSTERIOR_SNT_ENCODER} \
--debug_size=${DEBUG_SIZE} \
--optim_scheduler_name=${OPTIM_SCHEDULER_NAME} \
--warmup_proportion=${WARMUP_PROPORTION} \
--gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
--bert_model=${BERT_MODEL} \
--mask_pre_unaligned=${MASK_PRE_UNALIGNED} \
--result_folder=${RESULT_FOLDER} \
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
--dm_target_pos_bias=${DM_TARGET_BIAS} \
--dm_cat_bias=${DM_CAT_BIAS} \
--dm_sense_bias=${DM_SENSE_BIAS} \
--psd_target_bias=${PSD_TARGET_BIAS} \
--psd_sense_bias=${PSD_SENSE_BIAS} \
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
--summary_dir=${SUMMARY_DIR} \
--renyi_alpha=${RENYI_ALPHA}
'''

def get_args(config_file):
    CMD = 'source %s; echo "%s"' % (config_file, pargs)
    p = subprocess.Popen(CMD, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    #pargs_str = p.stdout.readlines()[0].strip()
    pargs_str = ''.join(p.stdout.readlines()).replace('\n',' ').strip()
    parser = get_parser()
    return parser.parse_args(pargs_str.split(' '))

def get_parser():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description='train')

    def comma_sep_float_list(comma_str):
        return [float(i_str) for i_str in comma_str.split(',') if i_str !='']

    def comma_sep_int_list(comma_str):
        return [int(i_str) for i_str in comma_str.split(',') if i_str !='']

    def comma_sep_str_list(comma_str):
        return [i_str for i_str in comma_str.split(',') if i_str !='']

    # command options
    command_settings = parser.add_argument_group('command settings')

    command_settings.add_argument(
        '--prepare',
        action='store_true',
        help='create the directories, prepare the vocabulary and embeddings')
    command_settings.add_argument(
        '--train',
        action='store_true',
        help='train the model')
    command_settings.add_argument(
        '--evaluate', action='store_true',
        help='evaluate the model on dev set')
    command_settings.add_argument(
        '--predict',
        action='store_true',
        help='predict the answers for test set with trained model')

    ## Data options, path settings
    path_settings = parser.add_argument_group('path settings')
    path_settings.add_argument('--suffix', default=".txt_pre_processed", type=str,
                        help="""suffix of files to combine""")
    path_settings.add_argument('--folder', default=allFolderPath, type=str ,
                        help="""the folder""")
    path_settings.add_argument('--build_folder', default="", type=str,
                        help="""the build folder for dict and rules""")
    path_settings.add_argument('--save_to', default="",
                        help="""folder to save""" )
    path_settings.add_argument('--result_folder', default="",
                        help="""folder to save generation results""" )
    path_settings.add_argument('--restore_from', default='',
                        help="""If training from a checkpoint then this is the
                        path to the pretrained model.""")
    path_settings.add_argument('--log_path', default="",help='path of the log file. If not set, logs are logged to console')
    path_settings.add_argument('--summary_dir', default="",help='path of the summary dir.')
    bert_settings = parser.add_argument_group('bert settings')
    bert_settings.add_argument('--bert_model', default="", type=str,
                               help="""the name for pretraind bert name, like bert-base-cased, this model is shared""")
    # https://github.com/google-research/bert#out-of-memory-issues
    bert_settings.add_argument('--max_bert_seq_length', default=64, type=int,
                               help="""max_bert_seq_length for wordpieces in bert_base when batchsize=32""")

    ## Model options
    model_settings = parser.add_argument_group('model settings')
    model_settings.add_argument('--frames',type=comma_sep_str_list,
                                default='amr',
                                help='choose the frames to parse, sperated with comma')
    model_settings.add_argument('--concept_snt_encoder', default="rnn", type=str,
                               help="""concept_snt_encoder, Model:Share:ARG, Model:rnn, bert-x-name, SHARE =[share, sep], ARG is hidden_size for rnn, for bert, it is top N layers, when N>=1, use trainable N layers, x <0, use static bert..""")
    model_settings.add_argument('--posterior_snt_encoder', default="rnn", type=str,
                               help="""posterior_snt_encoder, Model:Share:ARG, Model:rnn, bert-x-name, SHARE =[share, sep], ARG is hidden_size for rnn, for bert, it is top N layers, when N>=1, use trainable N layers, x <0, use static bert..""")

    model_settings.add_argument('--posterior_amr_encoder', default="rnn", type=str,
                               help="""posterior_amr_encoder, Model:id:args, Model:rnn, transformer""")

    model_settings.add_argument('--rel_snt_encoder', default="rnn", type=str,
                               help="""rel_snt_encoder, Model:Share:ARG, Model:rnn, bert-x-name, SHARE =[share, sep], ARG is hidden_size for rnn, for bert, it is top N layers, when N>=1, use trainable N layers, x <0, use static bert..""")
    model_settings.add_argument('--root_snt_encoder', default="rnn", type=str,
                               help="""root_snt_encoder, Model:Share:ARG, Model:rnn, bert-x-name, SHARE =[share, sep], ARG is hidden_size for rnn, for bert, it is top N layers, when N>=1, use trainable N layers, x <0, use static bert..""")

    model_settings.add_argument('--use_src_encs_for_posterior', type=bool, default=False,
                                help="whether use src encs as input for posterior")

    model_settings.add_argument('--use_src_encs_for_rel', type=bool, default=False,
                                help="whether use src encs as input for relation snt encoder")

    model_settings.add_argument('--use_src_encs_for_root', type=bool, default=False,
                                help="whether use src encs as input for root snt encoder"
    )
    model_settings.add_argument('--get_wiki', type=bool,default=True)
    model_settings.add_argument('--normalize_mod', type=bool,default=True)
    model_settings.add_argument('--get_sense', type=bool,default=True)
    model_settings.add_argument('--jamr', default=False, type=bool,
                        help="""wheather to use fixed alignment""")
    model_settings.add_argument('--cat_bias', type=int, default=1,
                        help='Wheather bias category')

    model_settings.add_argument('--lemma_bias', type=int, default=0,
                        help='Wheather bias lemma')

    model_settings.add_argument('--dm_target_pos_bias', type=int, default=1,
                        help='Wheather bias dm target pos')

    model_settings.add_argument('--dm_cat_bias', type=int, default=1,
                        help='Wheather bias dm cat')

    model_settings.add_argument('--dm_sense_bias', type=int, default=1,
                        help='Wheather bias dm sense')

    model_settings.add_argument('--psd_target_pos_bias', type=int, default=1,
                        help='Wheather bias psd target pos')

    model_settings.add_argument('--psd_sense_bias', type=int, default=1,
                        help='Wheather bias psd sense')

    model_settings.add_argument('--independent_posterior', type=int, default=1,
                        help='Wheather use independent_posterior')

    model_settings.add_argument('--train_posterior', type=int, default=1,
                        help='keep training posterior')

    model_settings.add_argument('--alpha_dropout', type=float, default=0.1,
                        help='unk with alpha for alpha dropout ')

    model_settings.add_argument('--initialize_word', type=bool, default=True,
                        help='Wheather initialize_lemma')

    #layers
    model_settings.add_argument('--rel_enlayers', type=int, default=2,
                        help='Number of layers in the rel LSTM encoder/decoder')
    model_settings.add_argument('--root_enlayers', type=int, default=1,
                        help='Number of layers in the root LSTM encoder/decoder')
    model_settings.add_argument('--txt_enlayers', type=int, default=1,
                        help='Number of layers in the concept LSTM encoder/decoder')
    model_settings.add_argument('--amr_enlayers', type=int, default=1,
                        help='Number of layers in the amr LSTM encoder/decoder')

    model_settings.add_argument('--txt_rnn_size', type=int, default=512)
    model_settings.add_argument('--rel_rnn_size', type=int, default=512,
                        help='Number of hidden units in the rel/root LSTM encoder/decoder')
    model_settings.add_argument('--amr_rnn_size', type=int, default=200)
    model_settings.add_argument('--char_encoder_config', type=str, default='{"char_encoder_type": "Cnn"}')
    model_settings.add_argument('--rel', type=float, default=1.0,
                        help='Wheather train relation/root identification')

    #dimensions
    model_settings.add_argument('--word_dim', type=int, default=300,
                        help='Word embedding sizes')
    model_settings.add_argument('--dim', type=int, default=200,
                        help='lemma/high embedding sizes')
    model_settings.add_argument('--pos_dim', type=int, default=32,
                        help='Pos embedding sizes')
    model_settings.add_argument('--char_dim', type=int, default=64,
                        help='Char embedding sizes')
    model_settings.add_argument('--ner_dim', type=int, default=16,
                        help='Ner embedding sizes')
    model_settings.add_argument('--cat_dim', type=int, default=32,
                        help='category embedding sizes')
    model_settings.add_argument('--rel_dim', type=int, default=200,
                        help='mixed amr node and text representation dimension')
    # parser.add_argument('--residual',   action="store_true", , type=bool, default=True
    #                     help="Add residual connections between RNN layers.")
    model_settings.add_argument('--brnn', type=bool, default=True,
                        help='Use a bidirectional encoder')

    model_settings.add_argument('--gumbel',type=bool, default=True, help='whether using gumbel softmax')

    model_settings.add_argument('--sink', type=int, default=10,
                        help='steps of sinkhorn procedure')

    model_settings.add_argument('--sink_t', type=float, default=1,
                        help='gumbel-sinkhorn temperature')

    model_settings.add_argument('--prior_t', type=float, default=5,
                        help='prior tempeture for gumbel-sinkhorn')

    model_settings.add_argument('--sink_re', type=float, default=10,
                        help='gumbel-sinkhorn finite step regularzor penalizing non double-stochaticity')

    model_settings.add_argument('--mask_pre_unaligned', type=float, default=1.0e6,
                        help='mask pre_unaligned alignments')

    model_settings.add_argument('--emb_independent', type=int, default=1,
                        help="""wheather relation system use independent embedding""")

    ## Optimization options
    training_settings = parser.add_argument_group('training settings')
    training_settings.add_argument('--weight_decay', type=float, default=0.00001,
                        help='l2 weight_decay')
    training_settings.add_argument('--retrain_all', type=bool, default=False,
                        help='wheather to train all parameters. useful when reloading model for train')

    training_settings.add_argument('--batch_size', type=int, default=64,
                        help='Maximum batch size')
    training_settings.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    training_settings.add_argument('--start_epoch', type=int, default=1,
                        help='The epoch from which to start')

    training_settings.add_argument('--optim', default='adam',
                                   help="Optimization method. [sgd|adagrad|adadelta|adam|AdamW|BertAdam]")
    training_settings.add_argument('--optim_json_configs', default='',
                                   help="""Optimization configs for param patterns in json format, key is a list of patteren, value is a dict of params options in pytorch""")
    training_settings.add_argument('--learning_rate', type=float, default=0.001,
                        help="""Starting learning rate. If adagrad/adadelta/adam is
                        used, then this is the global learning rate. Recommended
                        settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.1""")
    training_settings.add_argument('--max_grad_norm', type=float, default=10,
                        help="""If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm""")

    training_settings.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                   help="""use gradient accumulation per N steps, default:1, means not use""")

    training_settings.add_argument('--warmup_proportion', type=float, default=-1,
                                   help="""Proportion of training to perform linear learning
                                   rate warmup for. E.g., 0.1 = 10%% of training.""")

    training_settings.add_argument('--optim_scheduler_name', type=str, default="none",
                                   help="""optim scheduler name to use: """
                                   + """ """.join([name for name in SCHEDULES.keys() if name]))

    training_settings.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability; applied between LSTM stacks and some MLPs.')

    training_settings.add_argument('--debug_size', type=int, default=-1,
                                   help="""use a portion of training data for training, debug_size=-1, means no debug,
                                   when debug_size > 0, it means training and test on both the same portation of examples""")

    training_settings.add_argument('--learning_rate_decay', type=float, default=0.98,
                        help="""Decay learning rate by this much if (i) perplexity
                        does not decrease on the validation set or (ii) epoch has
                        gone past the start_decay_at_limit""")

    training_settings.add_argument('--start_decay_at', default=5,
                        help="Start decay after this epoch")

    training_settings.add_argument('--renyi_alpha', type=float,default=.5,
                        help="parameter of renyi_alpha relaxation, "
                             "which is alternative to hierachical relaxation in the paper")
    # GPU
    parser.add_argument('--gpus',type=comma_sep_int_list, default="-2",
                        help="Use ith gpu, if -1 then load in cpu, -2 then load by the outside gpu schudler")  #training probably cannot work on cpu
    parser.add_argument('--from_gpus',type=comma_sep_int_list, default="",
                        help="model load from which gpu, must be specified when current gpu id is different from saving time")

    parser.add_argument('--log_per_epoch', type=int, default=10,
                        help="Log stats at this interval.")

    return parser
