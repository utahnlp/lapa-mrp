#!/bin/bash
# source envrironment variables
. ./env.sh

pushd $ROOT_DIR/Expt/mrp_scripts/commands/
model_name=$2
gpuid=$3
frame=$4
echo "load model"${model_name}
./test_prebuild_gpuid.sh $1 ${model_name} $gpuid
./mtool.sh $1 $frame
popd
