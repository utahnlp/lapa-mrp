#!/bin/bash

# source env.sh
. ./env.sh

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd $ROOT_DIR
python utility/amr.py
popd

