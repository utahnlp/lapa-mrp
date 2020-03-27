#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# the root dir of this code rep
ROOT_DIR=$DIR/../../../
MRP_EXPT_DIR=$ROOT_DIR/Expt/
EXP_ROOT_DIR=$ROOT_DIR/Expt/workdirs/
MRP_DATA_DIR=$MRP_EXPT_DIR/data/mrp_data/
MRP_DATA_RO_DIR=$MRP_EXPT_DIR/data/mrp_data_ro/
CODE_BASE=$ROOT_DIR/../
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH

eval "$(pyenv init -)"
pyenv activate py2.7

# for conda prior to 4.6
# source activate python2
#conda activate python2
