#!/bin/bash
set -x
PIPE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# do preprocessing for AMR.
# with amr2eng.sh
$PIPE_DIR/amr2eng.sh || { echo 'amr2eng.sh failed' ; exit 1 ;}

# do rule_system build
echo "build rules"
$PIPE_DIR/rule_system_build.sh || { echo 'rule_system_build.sh failed' ; exit 1 ; }

echo "build data with pickle"
$PIPE_DIR/data_build.sh || { echo 'data_build.sh failed' ; exit 1 ; }

set +x
