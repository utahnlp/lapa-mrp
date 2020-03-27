#!/bin/bash

#SBATCH --job-name=test_pipe
#SBATCH --gres=gpu:1
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/test_pipe.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=40G
eval "$(pyenv init -)"
config_file=$1
model_name=$2
pushd $CODE_BASE/amr/Expt/amr-scripts/commands/
./test_prebuild.sh $config_file $model_name
./smatch.sh $config_file
popd
