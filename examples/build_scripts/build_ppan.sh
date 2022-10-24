#!/bin/bash

set -e

INSTALL_PREFIX=$1
ENVIRONMENT_NAME=$2

PACE_DIR=$(pwd)/../..
FV3NET_DIR=$INSTALL_PREFIX/fv3net

# module load necessary system software
module load conda
module load intel_compilers/2021.3.0

export MPICC=$(which mpicc)

CONDA_PREFIX=$INSTALL_PREFIX/conda
conda config --add pkgs_dirs $CONDA_PREFIX/pkgs
conda config --add envs_dirs $CONDA_PREFIX/envs

# enter the pace directory
cd $PACE_DIR

# create a conda environment with cartopy and its dependencies installed
conda create -c conda-forge -y --name $ENVIRONMENT_NAME python=3.8 matplotlib==3.5.2 cartopy==0.18.0

# enter the environment and update it
conda activate $ENVIRONMENT_NAME
pip3 install --upgrade --no-cache-dir pip setuptools wheel

# install the Pace dependencies, GT4Py, and Pace
pip3 install --no-cache-dir -r requirements_dev.txt -c constraints.txt

# clone fv3net
git clone https://github.com/ai2cm/fv3net.git $FV3NET_DIR

# install jupyter and ipyparallel
pip3 install --no-cache-dir \
       ipyparallel==8.4.1 \
       jupyterlab==3.4.4 \
       jupyterlab_code_formatter==1.5.2 \
       isort==5.10.1 \
       black==22.3.0

# install vcm
python3 -m pip install $FV3NET_DIR/external/vcm
