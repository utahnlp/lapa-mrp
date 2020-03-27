#!/bin/bash

#SBATCH --job-name=full_pipe
#SBATCH --gres=gpu:1
#SBATCH --output=/home/utah/jiecao/dgx_jobs/mrp/full_pipe.txt
#SBATCH --ntasks=1
#SBATCH --time=30:40:00
#SBATCH --mem=30G
config_file=$1
frame=$2

pushd $CODE_BASE/mrp/Expt/mrp_scripts/commands/
echo "assign gpus ids:"$CUDA_VISIBLE_DEVICES
./full_pipe.sh ${config_file} ${frame}
popd
