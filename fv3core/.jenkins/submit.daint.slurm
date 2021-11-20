#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=<NAME>
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --output=<OUTFILE>
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=cscsci


########################################################

set -x
export OMP_NUM_THREADS=1
export CRAY_CUDA_PROXY=0

<CMD>

########################################################
