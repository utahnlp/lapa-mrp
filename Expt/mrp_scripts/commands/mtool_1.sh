#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$ROOT_DIR/utility/mtool/

pushd $EVAL_DIR
./main.py --read mrp -tt --score mrp --gold $1 $2 2>&1 &> dev_ucca_mtool.log
popd

#conda deactivate
#pyenv deactivate 2.7.12
