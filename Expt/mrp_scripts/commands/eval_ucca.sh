#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$ROOT_DIR/utility/mtool/
GOLD_DIR=/home/ubuntu/git-workspace/mrp/Expt/data/mrp_data/ucca_retok/
PRED_DIR=/home/ubuntu/git-workspace/self-attentive-parser/Expt/training_ucca_retok/

pushd $EVAL_DIR
./main.py --read mrp -tt --score mrp --gold $GOLD_DIR/dev/dev.mrp_ucca $PRED_DIR/dev_out/dev.out_tree.ucca &> $PRED_DIR/dev_out/mtool_dev_ucca.log
popd


