#!/bin/bash

# Example bash script to build Pace to run locally outside of a container, assuming Pace is cloned into $HOME

set -e -x

# module load necessary system software
module load gcc/11.3.0
module load openmpi
module load cuda/10.2
module load python/3.8.13

# download boost and add to path:
mkdir src
cd src
wget https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
tar -xzf boost_1_79_0.tar.gz
mkdir -p boost_1_79_0/include
mv boost_1_79_0/boost boost_1_79_0/include/.
setenv BOOST_ROOT $HOME/src/boost_1_79_0

# go into pace, update submodules and download test data:
cd $HOME/pace
git submodule update --init
cd fv3core
make get_test_data
cd ..

# create python venv for pace and install the pace requirements:
# you may need to set the MPICC environment variable to resolve includes of mpi.h
python3 -m venv venv
source ./venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r ./requirements.txt -c ./constraints.txt

# install correct cudapy
pip3 install cupy-cuda102==7.7.0

# install gt4py
pip3 install --no-cache-dir -c ./constraints.txt "$PYHOME/PACE/pace/external/gt4py"
python3 -m gt4py.gt_src_manager install --major-version 2

# install pace components into the venv:
pip3 install -e ./pace-util
pip3 install -e ./fv3core
pip3 install -e ./fv3gfs-physics
pip3 install -e ./stencils
pip3 install -e ./dsl
pip3 install -e ./driver

#You should now be able to run the tests successfully:
python -m pytest --data_path=fv3core/test_data/8.1.1/c12_6ranks_standard/dycore/ \
     -v -s -rsx --disable-warnings --exitfirst --backend=numpy --which_modules=Fillz \
     fv3core/tests/savepoint/test_translate.py --print_failures \
     --threshold_overrides_file=fv3core/tests/savepoint/translate/overrides/standard.yaml
