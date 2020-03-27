#! /bin/bash
# $1 is the configs folder path
set -x
configs_folder=$1
configs=`find ${configs_folder} -name "*.sh"`

for i in $configs; do
  sed -i 's/_64_padding/_17_padding/g' $i
done

set +x
