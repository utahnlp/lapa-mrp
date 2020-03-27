#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sub_name=$1

BUILD_DIR=$ROOT_DIR/Expt/data/mrp_data/$sub_name/

### preprocessing
pargs="
--companion_suffix=.mrp_conllu \
--build_folder=$BUILD_DIR \
--frame=amr \
--token_combine=x \
"

pushd $ROOT_DIR
python src/preprocessing.py $pargs &> $ROOT_DIR/Expt/pre_logs/amr_preprocessing_${sub_name}.log
popd

