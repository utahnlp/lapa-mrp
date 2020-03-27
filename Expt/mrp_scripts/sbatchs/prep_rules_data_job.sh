#!/bin/bash

#SBATCH --job-name=prep_rules_data
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/prep_rules_data.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=40G
eval "$(pyenv init -)"
pyenv activate py3.6.5_torch
pushd $CODE_BASE/amr/Expt/amr-scripts/commands/
./prep_rules_data.sh $1
popd
