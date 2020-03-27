#!/bin/bash

#SBATCH --job-name=pipe
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/pre_pipe.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=40G
eval "$(pyenv init -)"
pyenv activate py3.6.5_torch
pushd $CODE_BASE/amr/Expt/amr-scripts/commands/
./pre_pipe_alignments.sh 
popd
