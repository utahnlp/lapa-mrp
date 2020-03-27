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
--training_ids="/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/amr_splits_new/training/training.mrp_amr.ids" \
--dev_ids="/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/amr_splits_new/dev/dev.mrp_amr.ids" \
--test_ids="/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/amr_splits_new/test/test.mrp_amr.ids" \
"

pushd $ROOT_DIR
python utility/mrp_utils.py $pargs &> $ROOT_DIR/Expt/pre_logs/mrp_utils_amr_conllu.log
popd

