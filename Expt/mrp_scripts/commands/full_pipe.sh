#!/bin/bash
# source envrironment variables
. ./env.sh

config_file=$1
frame=$2

pushd $ROOT_DIR/Expt/mrp_scripts/commands/
./train.sh ${config_file}
model_name=gpus_${CUDA_VISIBLE_DEVICES}valid_best.pt
echo "load model"${model_name}
./test_prebuild.sh ${config_file} ${model_name}
./mtool.sh ${config_file} ${frame}
popd
