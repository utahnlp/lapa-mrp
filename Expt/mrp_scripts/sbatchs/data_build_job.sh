#!/bin/bash

#SBATCH --job-name=data_build
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/data_build.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=40G
eval "$(pyenv init -)"
pyenv activate py3.6.5_torch
pushd $CODE_BASE/amr/Expt/amr-scripts/commands/
./data_build.sh $1
popd
