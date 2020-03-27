#!/bin/bash
# source envrironment variables
. ./env.sh

config_name=$1
model_name=$2
gpuid=$3
frame=$4

pushd $ROOT_DIR/Expt/mrp_scripts/commands/
./train_restore_gpuid.sh $config_name $model_name $gpuid
echo "load model"${model_name}
./test_prebuild_gpuid.sh $1 ${model_name} $gpuid
./mtool.sh $1 $frame
popd
