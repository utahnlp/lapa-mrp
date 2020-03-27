#!/bin/bash

#SBATCH --job-name=full_pipe
#SBATCH --gres=gpu:1
#SBATCH --output=/home/utah/jiecao/dgx_jobs/mrp/full_pipe.txt
#SBATCH --ntasks=1
#SBATCH --time=30:40:00
#SBATCH --mem=30G
config_name=$1
gpuid=$2
frame=$3

pushd $CODE_BASE/mrp/Expt/mrp-scripts/commands/
echo "assign gpus ids:"$CUDA_VISIBLE_DEVICES
./full_pipe_gpuid.sh $config_name $gpuid $frame
popd
