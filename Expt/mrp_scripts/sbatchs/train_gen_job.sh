#!/bin/bash

#SBATCH --job-name=train_dev
#SBATCH --gres=gpu:1
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/train_dev.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=30G
eval "$(pyenv init -)"
pyenv activate py3.6.5_torch0.2.0
pushd $CODE_BASE/amr/Expt/amr-scripts/commands/
./train_dev.sh $1 
popd
