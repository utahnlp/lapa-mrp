#!/bin/bash
# source environment
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sub_name=$1

BUILD_DIR=$ROOT_DIR/Expt/data/mrp_data/$sub_name/

### TRAIN for task1
pargs="--threshold=5 \
--jamr=0 \
--skip=0 \
--merge_common_dicts=0 \
--suffix=.mrp_amr \
--companion_suffix=.mrp_conllu_pre_processed \
--build_folder=${BUILD_DIR} \
--bert_model=bert-large-cased \
"
# to lower case toggle this
#--do_lower_case=x

pushd $ROOT_DIR
python src/amr_data_build.py $pargs &> $ROOT_DIR/Expt/pre_logs/amr_data_build_${sub_name}.log
popd

