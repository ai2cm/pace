#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=standalone
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=run.daint.out
#SBATCH --hint=nomultithread
#SBATCH --time=00:08:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=debug

########################################################

# source an venv with cupy, mpi4py, fv3core installed
# load corresponding cuda module to cupy version (10+)

module load nvidia-nsight-systems/2021.1.1.66-6c5c5cb

########################################################

# To be launched from ./fv3core, will copy the results in ./profiling_results

set -x
export OMP_NUM_THREADS=12
export CRAY_CUDA_PROXY=0
export PYTHONOPTIMIZE=TRUE
export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:$PYTHONPATH
srun nsys profile --force-overwrite=true \
  -o ./profiling_results/%h.%q{SLURM_NODEID}.%q{SLURM_PROCID}.qdstrm \
  --trace=cuda,mpi,nvtx --mpi-impl=mpich \
  --stats=true \
  python ./profiler/external_profiler.py \
  examples/standalone/runfile/dynamics.py /project/s1053/fv3core_serialized_test_data/7.2.5/c128_6ranks_baroclinic/ 2 gtcuda profiling

########################################################
