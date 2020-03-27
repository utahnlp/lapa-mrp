#!/bin/bash
set -x
PIPE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sub_name=$1

# do rule_system build
echo "build dm rules"
$PIPE_DIR/dm_rule_system_build.sh ${sub_name} || { echo 'rule_system_build_dm.sh ${sub_name} failed' ; exit 1 ; }

echo "build eds dm with pickle"
$PIPE_DIR/dm_data_build.sh ${sub_name} || { echo 'data_build_dm.sh ${sub_name} failed' ; exit 1 ; }

set +x
