#!/bin/bash
# source environment
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sub_name=$1

BUILD_DIR=$ROOT_DIR/Expt/data/mrp_data/$sub_name/

### TRAIN for task1
pargs="
--suffix=.mrp_dm \
--companion_suffix=.mrp_conllu_pre_processed \
--build_folder=${BUILD_DIR} \
"

pushd $ROOT_DIR
python utility/dm_utils/test_mrp_dm.py $pargs 2>&1 &> $ROOT_DIR/Expt/pre_logs/test_mrp_dm_${sub_name}.log
popd

