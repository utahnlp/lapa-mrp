#!/bin/bash

#SBATCH --job-name=te_test
#SBATCH --gres=gpu:1
#SBATCH --output=/home/utah/jiecao/dgx_jobs/train_eval_test.txt
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --mem=15G
eval "$(pyenv init -)"
pyenv activate py2.7_tf1.4
pushd $CODE_BASE/psyc/Expt/psyc-scripts/commands/
./train_eval_test.sh $1 
popd
