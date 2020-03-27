#!/bin/bash

# source envoronment variables
. ./env2.sh

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

AMR_EVAL_DIR=$CODE_BASE/amr-evaluation-tool-enhanced/

pushd $AMR_EVAL_DIR
#./evaluation.sh $RESULT_FOLDER/training.combined.txt_without_jamr_processed_generate $FOLDER/../../amrs/split/training/training.combined.origin &> $RESULT_FOLDER/training.combined.txt_without_jamr_smatch
./evaluation.sh $RESULT_FOLDER/dev.txt_without_jamr_processed_generate $BUILD_FOLDER/dev/dev.amr_ori &> $RESULT_FOLDER/dev.txt_without_jamr_smatch
./evaluation.sh $RESULT_FOLDER/test.txt_without_jamr_processed_generate $BUILD_FOLDER/test/test.amr_ori &> $RESULT_FOLDER/test.txt_without_jamr_smatch
popd

pyenv deactivate
