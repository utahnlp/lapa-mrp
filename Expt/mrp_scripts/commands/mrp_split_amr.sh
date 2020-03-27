#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

input_folder=/home/ubuntu/git-workspace/mrp_data/mrp/2019/training/amr/

### preprocessing
pargs="
--suffix=.mrp \
--input_folder=$input_folder \
--training_ids="/home/ubuntu/git-workspace/mrp_data/mrp/2019/ori_amr_training.ids" \
--dev_ids="/home/ubuntu/git-workspace/mrp_data/mrp/2019/ori_amr_dev.ids" \
--test_ids="/home/ubuntu/git-workspace/mrp_data/mrp/2019/ori_amr_test.ids" \
"

pushd $ROOT_DIR
python utility/mrp_utils.py $pargs &> $ROOT_DIR/Expt/pre_logs/mrp_utils_amr.log
popd

