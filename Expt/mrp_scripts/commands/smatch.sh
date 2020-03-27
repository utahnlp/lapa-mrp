#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [[ "$1" = /* ]]; then
  config_data=$1
else
  config_data=$DIR/$1
fi

### configurate data directory
if [ ! -f ${config_data} ]; then
  echo "${config_data} doesn't exist"
  exit $?
else
  . ${config_data}
  echo "run ${config_data}"
fi

### CHECK WORK & DATA DIR
if [ ! -e ${EXP_DIR} ]; then
  echo "original model not exist"
  exit -1
fi

AMR_EVAL_DIR=$ROOT_DIR/utility/mtool/

pushd $AMR_EVAL_DIR
./main.py --read mrp --score smatch --gold $BUILD_FOLDER/dev/dev.mrp_amr  $RESULT_FOLDER/dev.mrp_conllu_mrp_generate &> $RESULT_FOLDER/dev.mrp_conllu_mrp_generate_smatch
./main.py --read mrp --score smatch --gold $BUILD_FOLDER/test/test.mrp_amr  $RESULT_FOLDER/test.mrp_conllu_mrp_generate &> $RESULT_FOLDER/test.mrp_conllu_mrp_generate_smatch
popd

#conda deactivate
#pyenv deactivate 2.7.12
