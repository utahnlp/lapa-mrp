#!/bin/bash
# source environment
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
snt_out=${1}_snt
### TRAIN for task1
pargs="
--input=$1
--extract_snt=$snt_out
"

pushd $ROOT_DIR
python utility/ucca_utils/ptb2ucca.py $pargs 2>&1 &> $ROOT_DIR/Expt/pre_logs/extract_ucca_snt.log
popd

