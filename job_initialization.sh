#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=Job
#SBATCH --ntasks="6"
#SBATCH --hint=multithread
#SBATCH --ntasks-per-core=2
#SBATCH --ntasks-per-node="24"
#SBATCH --cpus-per-task=1
#SBATCH --output=Job.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=cscsci


########################################################

set -x
export OMP_NUM_THREADS=1

srun python3 /scratch/snx3000/ewu/pace/.jenkins/..//driver/examples/baroclinic_init.py /scratch/snx3000/ewu/pace/.jenkins/driver_configs/baroclinic_c192_6ranks.yaml

########################################################
