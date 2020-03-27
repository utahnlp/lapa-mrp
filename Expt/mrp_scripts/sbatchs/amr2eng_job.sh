#!/bin/bash

#SBATCH --job-name=amr2eng
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/amr2eng.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=40G
eval "$(pyenv init -)"
pyenv activate py3.6.5_torch
pushd $CODE_BASE/amr/Expt/scripts/commands/
./amr2eng.sh
popd
