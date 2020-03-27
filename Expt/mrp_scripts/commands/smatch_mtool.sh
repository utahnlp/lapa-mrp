#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AMR_EVAL_DIR=$ROOT_DIR/utility/mtool/

pushd $AMR_EVAL_DIR
./main.py --read mrp --score smatch --gold $1 $2 &> $DIR/smatch.log
popd

#conda deactivate
#pyenv deactivate 2.7.12
