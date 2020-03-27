#!/bin/bash

# source envoronment variables
. ./env.sh

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

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EVAL_DIR=$ROOT_DIR/utility/mtool/

pushd $EVAL_DIR
./main.py --read mrp -ttt --score mrp --gold ${BUILD_FOLDER}/dev/dev.mrp_dm ${BUILD_FOLDER}/dev/dev.mrp_dm &> ${EXP_RESULTS}/mtool_dev_gold.log
popd

#conda deactivate
#pyenv deactivate 2.7.12
