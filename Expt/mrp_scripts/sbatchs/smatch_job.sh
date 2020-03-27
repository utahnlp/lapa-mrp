#!/bin/bash

#SBATCH --job-name=smatch
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/smatch.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=20G

pushd $CODE_BASE/amr/Expt/amr-scripts/commands/
./smatch.sh $1
popd
