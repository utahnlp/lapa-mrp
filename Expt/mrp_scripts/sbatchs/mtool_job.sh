#!/bin/bash

#SBATCH --job-name=mtool_pipe
#SBATCH --output=/home/utah/jiecao/dgx_jobs/mrp/full_pipe.txt
#SBATCH --ntasks=1
#SBATCH --time=30:40:00
#SBATCH --mem=30G
config_file=$1
frame=$2

pushd $CODE_BASE/mrp/Expt/mrp_scripts/commands/
./mtool.sh ${config_file} ${frame}
popd
