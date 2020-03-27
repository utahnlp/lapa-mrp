#! /bin/bash
# $1 is the configs folder path
# $2 is the command folder path
set -x
configs_folder=$1
commands_folder=$2
configs=`find ${configs_folder} -name "*.sh"`
commands=`find ${commands_folder} -name "train.sh" -o -name "train_restore.sh" -o -name "test_prebuild.sh" -o -name "parse_amr.sh" -o -name "train_gpuid.sh" -o name "test_prebuild_gpuid.sh"`
ARG_COMMENT="# Optimization Methods, default is adam, support sgd,adagrad, adadelta, adam"
arg_name_lc="optim"
ARG_NAME="OPTIM"

#ARG_VALUE=transformer:pamr:\{\"hidden_dim\":256,\"projection_dim\":256,\"feedforward_hidden_dim\":200,\"num_layers\":2,\"num_attention_heads\":4\}
ARG_VALUE=1
for i in $configs; do
  if grep -Fq "${ARG_NAME}" $i; then
    #sed -i "/^${ARG_COMMENT}/d" $i
    sed -i "/${ARG_COMMENT}/d" $i
    sed -i "/^${ARG_NAME}/d" $i
  else
    printf "args in $i already has already removed"
  fi
done

# declare -a scripts=( 'prepare.sh' 'train.sh' 'dev.sh' 'train_dev.sh' )

for j in ${commands[@]}; do
  if grep -Fq "${arg_name_lc}" $j; then
    sed -i "/--${arg_name_lc}/d" $j
  else
    printf "args in $j already removed"
  fi
done

set +x
