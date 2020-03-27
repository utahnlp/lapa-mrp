#!/bin/bash

#SBATCH --job-name=rule_system_build
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/rule_system_build.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=40G
eval "$(pyenv init -)"
pyenv activate py3.6.5_torch
pushd $CODE_BASE/amr/Expt/scripts/commands/
./rule_system_build.sh $1
popd
