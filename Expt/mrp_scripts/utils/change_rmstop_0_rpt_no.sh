#!/bin/bash
set -x
config_folder=$1
sub_folder=$config_folder/rmstop_0_rpt_no/
mkdir -p $sub_folder
configs=`grep -l "TOKEN_KEY_TO_USE=tokenized_utterance" $config_folder/*.sh`
for i in $configs; do
  sed -i 's/FILE1=\$DATA_DIR\/\$/FILE1=\$DATA_DIR\/prep_data\/rmstop_0_rpt_no\/\$/g' $i
  mv $i $sub_folder
done
set +x
