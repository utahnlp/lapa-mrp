#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

input_folder=/home/ubuntu/git-workspace/mrp_data/mrp/2019/evaluation/

### preprocessing
pargs="
--suffix=.mrp_conllu \
--input_folder=$input_folder \
"

pushd $ROOT_DIR
python utility/mrp_conllu_utils.py $pargs &> $ROOT_DIR/Expt/pre_logs/mrp_utils_conllu.log
popd

