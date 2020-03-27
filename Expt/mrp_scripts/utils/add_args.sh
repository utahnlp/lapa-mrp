#! /bin/bash
# $1 is the configs folder path
# $2 is the command folder path
set -x
configs_folder=$1
commands_folder=$2
configs=`find ${configs_folder} -name "*.sh"`
commands=`find ${commands_folder} -name "train.sh" -o -name "train_restore.sh" -o -name "test_prebuild.sh" -o -name "parse_amr.sh" -o -name "train_gpuid.sh" -o -name "test_prebuild_gpuid.sh"`
#commands=`find ${commands_folder} -name "train.sh" -o -name "train_restore.sh" -o -name "test_prebuild.sh"`
ARG_COMMENT="# char dim"
arg_name_lc="char_dim"
ARG_NAME="CHAR_DIM"
#ARG_VALUE=transformer:pamr:\{\"hidden_dim\":256,\"projection_dim\":256,\"feedforward_hidden_dim\":200,\"num_layers\":2,\"num_attention_heads\":4\}
ARG_VALUE=64

for i in $configs; do
  if grep -Fq "${ARG_COMMENT}" $i; then
    printf "$i already has already taken effect"
  else
    printf "${ARG_COMMENT}\n${ARG_NAME}=${ARG_VALUE}\n" >> $i
  fi
done

for j in ${commands[@]}; do
  if grep -Fq "${ARG_NAME}" $j; then
    printf "$j already take effect"
  else
    sed -i "/pargs=\"/a --${arg_name_lc}=\$\{${ARG_NAME}\} \\\\" $j
  fi
done

set +x
