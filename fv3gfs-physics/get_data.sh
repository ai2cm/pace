#!/bin/bash

# this scripts downloads test data for the physics standalone

# echo on
set -x


# remove preexisting data directory
test -d ./data
/bin/rm -rf data

# get data
wget --quiet "ftp://ftp.cscs.ch/in/put/abc/cosmo/fuo/physics_standalone/GFSPhysicsDriver/c12_6ranks_baroclinic_dycore_microphysics.tar.gz"
test -f c12_6ranks_baroclinic_dycore_microphysics.tar.gz || exit 1
tar -xvf c12_6ranks_baroclinic_dycore_microphysics.tar.gz || exit 1
/bin/rm -f c12_6ranks_baroclinic_dycore_microphysics.tar.gz 2>/dev/null

# done
