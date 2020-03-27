#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

input_folder=/home/ubuntu/git-workspace/mrp_data/mrp/2019/companion/

### preprocessing
pargs="
--follow_ids_only=x \
--suffix=.mrp_conllu \
--input_folder=$input_folder \
--training_ids="/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/ucca/training/training.mrp_ucca.ids" \
--dev_ids="/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/ucca/dev/dev.mrp_ucca.ids" \
--test_ids="/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/ucca/test/test.mrp_ucca.ids" \
"

pushd $ROOT_DIR
python utility/mrp_utils.py $pargs &> $ROOT_DIR/Expt/pre_logs/mrp_utils_ucca_conllu.log
popd

