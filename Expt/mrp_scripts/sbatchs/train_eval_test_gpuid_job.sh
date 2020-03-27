#!/bin/bash

#SBATCH --job-name=te_test
#SBATCH --output=/home/utah/jiecao/dgx_jobs/train_eval_test_gpuid.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=15G
eval "$(pyenv init -)"
pyenv activate py2.7_tf1.4
pushd $CODE_BASE/psyc/Expt/psyc-scripts/commands/
./train_eval_test_gpuid.sh $1 $2
popd
