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
--training_ids="/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/dm/training/training.mrp_dm.ids" \
--dev_ids="/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/dm/dev/dev.mrp_dm.ids" \
--test_ids="/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/dm/test/test.mrp_dm.ids" \
"

pushd $ROOT_DIR
python utility/mrp_utils.py $pargs &> $ROOT_DIR/Expt/pre_logs/mrp_utils_dm_conllu.log
popd

