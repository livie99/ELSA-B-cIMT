#!/bin/bash
#SBATCH --account elsa-brazil
#SBATCH --mem 32g
#SBATCH --cpus-per-task=16
#SBATCH --partition gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time 2:00:00

cd /home/livieymli/elsa-brazil/ELSA-Brazil/
export OMP_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export TORCH_DISTRIBUTED_DISABLE_PARAMETER_CHECK=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
/home/livieymli/miniforge3/envs/cu312/bin/python regression_mlp.py -i 'crfs'