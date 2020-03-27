#!/bin/bash
# source envrironment variables
. ./env.sh

pushd $ROOT_DIR/Expt/mrp_scripts/commands/
model_name=gpus_2valid_best.pt
echo "load model"${model_name}
./test_prebuild2.sh $1 ${model_name}
./smatch.sh $1
popd
