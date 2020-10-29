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
for package in /external/*
do
    if [[ -f $package/setup.py ]]
    then
        echo "Setting up $package"
        pip install -e "$package" -c /constraints.txt
    fi
done
pip install -e /fv3core -c /constraints.txt

exec "$@"
