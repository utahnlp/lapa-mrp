#!/bin/bash
set -x
PIPE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sub_name=$1

# do rule_system build
#echo "preprocessing amr inputs"
#$PIPE_DIR/preprocessing_amr.sh ${sub_name} || { echo 'preprocessing_amr.sh ${sub_name} failed' ; exit 1 ; }

echo "build psd rules"
$PIPE_DIR/psd_rule_system_build.sh ${sub_name} || { echo 'psd_rule_system_build.sh ${sub_name} failed' ; exit 1 ; }

echo "build psd data with pickle"
$PIPE_DIR/psd_data_build.sh ${sub_name} || { echo 'psd_data_build.sh ${sub_name} failed' ; exit 1 ; }

set +x
