#!/bin/bash
# source envrironment variables
. ./env.sh

config_name=$1
gpuid=$2
frame=$3

pushd $ROOT_DIR/Expt/mrp_scripts/commands/
./train_gpuid.sh $config_name $gpuid
model_name=gpus_${gpuid}valid_best.pt
echo "load model"${model_name}
./test_prebuild_gpuid.sh $1 ${model_name} $gpuid
./mtool.sh $1 $frame
popd
