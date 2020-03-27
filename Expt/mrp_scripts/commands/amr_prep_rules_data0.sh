#!/bin/bash
set -x
PIPE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sub_name=$1

# do rule_system build
#echo "preprocessing amr inputs"
#$PIPE_DIR/preprocessing_amr.sh ${sub_name} || { echo 'preprocessing_amr.sh ${sub_name} failed' ; exit 1 ; }

echo "build amr rules"
$PIPE_DIR/amr_rule_system_build0.sh ${sub_name} || { echo 'rule_system_build_amr0.sh ${sub_name} failed' ; exit 1 ; }

echo "build amr data with pickle"
$PIPE_DIR/amr_data_build0.sh ${sub_name} || { echo 'data_build_amr0.sh ${sub_name} failed' ; exit 1 ; }

set +x
