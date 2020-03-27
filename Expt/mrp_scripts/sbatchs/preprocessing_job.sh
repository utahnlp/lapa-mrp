#!/bin/bash

#SBATCH --job-name=preprocessing
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/preprocessing.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=60G
eval "$(pyenv init -)"
pyenv activate py3.6.5_torch
pushd $CODE_BASE/amr/Expt/amr-scripts/commands/
./preprocessing.sh
popd
