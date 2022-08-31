#!/bin/bash

# Example bash script to build Pace to run bare-metal on Gaea's c4 cluster

set -e -x

# module load necessary system software
module rm PrgEnv-intel
module load PrgEnv-gnu
module rm gcc
module load gcc/10.3.0
module load boost/1.72.0
module load python/3.9

# set environment variables
export MPICC=$CC

# clone Pace and update submodules
git clone https://github.com/ai2cm/pace
cd pace
git submodule update --init --recursive

# create a conda environment for pace
conda create -y --name my_name python=3.8

# enter the environment and update it
conda activate my_name
pip3 install --upgrade pip setuptools wheel

# install the required dependencies
pip3 install -r requirements.txt -c constraints.txt

# install gt4py
pip3 install --no-cache-dir -c ./constraints.txt "$PWD/external/gt4py"
python3 -m gt4py.gt_src_manager install --major-version 2

# install pace
pip install -r requirements_dev.txt -c constraints.txt
