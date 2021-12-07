#!/bin/bash

set -e


if [[ -d /external/fv3gfs-fortran ]]
then
    # Re-compile the fortran and wrapper sources
    make -C /external/fv3gfs-fortran/FV3
    PREFIX=/usr/local make -C /external/fv3gfs-fortran/FV3 install
fi
# install /external python packages including wrapper
if [[ -d /external/fv3gfs-wrapper ]]
then
    CC=mpicc MPI=mpich make -C /external/fv3gfs-wrapper build
fi
pip install -e /external/pace-util -e /external/dsl -e /external/stencils -c constraints.txt
pip install -e /fv3core -c /constraints.txt

exec "$@"
