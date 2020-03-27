#!/bin/bash

# source envoronment variables
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sub_name=$1

BUILD_DIR=$ROOT_DIR/Expt/data/mrp_data/$sub_name/

pushd $ROOT_DIR
python utility/dm_utils/SEMIReader.py $pargs &> $ROOT_DIR/Expt/pre_logs/dm_semi_reader_${sub_name}.log
popd

