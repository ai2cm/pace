#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=pace
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=pace.out
#SBATCH --hint=nomultithread
#SBATCH --time=0:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=debug


########################################################

set -x
export OMP_NUM_THREADS=12
export CRAY_CUDA_PROXY=0
export PYTHONOPTIMIZE=TRUE
export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:$PYTHONPATH

########################################################

srun python examples/dyn_phy.py \
    ./test_data/c128_6ranks_baroclinic_dycore_microphysics \
    15 gtc:gt:gpu

# NSYS run of Pace
# module load nvidia-nsight-systems/2021.1.1.66-6c5c5cb
# srun nsys profile --force-overwrite=true \
#     -o ./profiling_results/pace_microph_gtcgtgpu_%h.%q{SLURM_NODEID}.%q{SLURM_PROCID}.qdstrm \
#     --trace=cuda,mpi,nvtx --mpi-impl=mpich \
#     --stats=true \
#     python /scratch/snx3000/fdeconic/performance_visualization/analysis/pywrapper.py --nvtx \
#         --config /scratch/snx3000/fdeconic/performance_visualization/analysis/config_examples/pace.json \
#         examples/dyn_phy.py \
#         ./test_data/c128_6ranks_baroclinic_dycore_microphysics/ \
#         15 gtc:cuda
        
########################################################
