#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILDENV_DIR=$SCRIPT_DIR/../../buildenv

VERSION=vcm_1.0
env_file=env.daint.sh
src_dir=$(pwd)

# module environment
source ${BUILDENV_DIR}/machineEnvironment.sh
source ${BUILDENV_DIR}/${env_file}

# echo commands and stop on error
set -e
set -x

dst_dir=${1:-${installdir}/venv/${VERSION}}
wheeldir=${2:-${installdir}/wheeldir}
save_wheel=${3: false}

# delete any pre-existing venv directories
if [ -d ${dst_dir} ] ; then
    /bin/rm -rf ${dst_dir}
fi

# setup virtual env
python3 -m venv ${dst_dir}
source ${dst_dir}/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade wheel

# installation of standard packages that are backend specific
if [ $save_wheel ]; then
    python3 -m pip wheel --wheel-dir=$wheeldir cupy Cython clang-format
fi
python3 -m pip install --find-links=$wheeldir cupy Cython clang-format

MPICC=cc pip install mpi4py==3.1.3

# deactivate virtual environment
deactivate

# echo module environment
echo "Note: this virtual env has been created on `hostname`."
cat ${BUILDENV_DIR}/${env_file} ${dst_dir}/bin/activate > ${dst_dir}/bin/activate~
mv ${dst_dir}/bin/activate~ ${dst_dir}/bin/activate


exit 0
