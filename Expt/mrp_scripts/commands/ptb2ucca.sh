#!/bin/bash
# source environment
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
snt_out=${2}_snt
### TRAIN for task1
pargs="
--input_ptb=$1
--input=$2
--extract_snt=$snt_out
"

pushd $ROOT_DIR
python utility/ucca_utils/ptb2ucca.py $pargs 2>&1 &> $ROOT_DIR/Expt/pre_logs/ptb2ucca.log
popd

