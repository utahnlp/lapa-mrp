#!/bin/bash
# source environment
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sub_name=$1

BUILD_DIR=$ROOT_DIR/Expt/data/mrp_data/$sub_name/

### TRAIN for task1
pargs="
--suffix=.mrp_eds \
--companion_suffix=.mrp_conllu_pre_processed \
--build_folder=${BUILD_DIR} \
"

pushd $ROOT_DIR
python utility/eds_utils/test_mrp_eds.py $pargs 2>&1 &> $ROOT_DIR/Expt/pre_logs/test_mrp_eds_${sub_name}.log
popd

