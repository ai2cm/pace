#!/bin/bash

INSTALL_PREFIX=$1
ENVIRONMENT_NAME=$2

PACE_DIR=$(pwd)/../..
FV3NET_DIR=$INSTALL_PREFIX/fv3net

module load conda
module load intel_compilers/2021.3.0

conda activate $ENVIRONMENT_NAME
export PYTHONPATH=$FV3NET_DIR/external/fv3viz:$PACE_DIR/external/gt4py/src
