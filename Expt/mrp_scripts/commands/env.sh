#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# the root dir of this code rep
ROOT_DIR=$DIR/../../../
MRP_EXPT_DIR=$ROOT_DIR/Expt/
EXP_ROOT_DIR=$ROOT_DIR/Expt/workdirs/
MRP_DATA_DIR=$MRP_EXPT_DIR/data/mrp_data/
MRP_DATA_RO_DIR=$MRP_EXPT_DIR/data/mrp_data_ro/
CODE_BASE=$ROOT_DIR/../
export PYTHONPATH=$ROOT_DIR:$ROOT_DIR/utility/mtool/:$PYTHONPATH

eval "$(pyenv init -)"
pyenv activate py3.6.5_torch

# for conda prior to 4.6
# source activate py3.6.5_torch
