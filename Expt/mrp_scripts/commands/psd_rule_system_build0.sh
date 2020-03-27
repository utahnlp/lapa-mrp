#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sub_name=$1

BUILD_DIR=$ROOT_DIR/Expt/data/mrp_data/$sub_name/

### build rules
pargs="--threshold=0 \
--suffix=.mrp_psd \
--companion_suffix=.mrp_conllu_pre_processed \
--build_folder=$BUILD_DIR \
"

pushd $ROOT_DIR
python src/psd_rule_system_build.py $pargs &> $ROOT_DIR/Expt/pre_logs/psd_rule_system_build0_${sub_name}.log
popd

