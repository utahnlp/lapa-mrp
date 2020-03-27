#!/bin/bash

#SBATCH --job-name=full_pipe
#SBATCH --gres=gpu:1
#SBATCH --output=/home/utah/jiecao/dgx_jobs/amr/full_pip.txt
#SBATCH --ntasks=1
#SBATCH --time=80:40:00
#SBATCH --mem=40G
eval "$(pyenv init -)"
pyenv activate py3.6.5_torch0.2.0
pushd $CODE_BASE/amr/Expt/amr-scripts/commands/
echo "assign gpus ids:"$CUDA_VISIBLE_DEVICES
./train_restore.sh $1 $2
model_name=gpus_${CUDA_VISIBLE_DEVICES}valid_best.pt
echo "load model"${model_name}
./test_prebuild.sh $1 ${model_name}
./smatch.sh $1
popd
