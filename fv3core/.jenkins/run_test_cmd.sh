#!/usr/bin/env bash

set -e -x

# Read arguments
command="$1"
echo $command
job_name="job-$2"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

. $SCRIPT_DIR/daint_tools.sh

if [ "${target}" == "gpu" ] ; then
    # we only run this on HPC
    set +e
    module load cray-python
    module load pycuda
    module load daint-gpu
    set -e
    gt4py_extras="[cuda]"
    NVCC_PATH=$(which nvcc)
    export CUDA_HOME=$(echo $NVCC_PATH | sed -e "s/\/bin\/nvcc//g")
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
else
    gt4py_extras=""
fi
set +e
module load gridtools/1_1_3
module load gridtools/2_1_0_b
module load /project/s1053/install/modulefiles/gcloud/303.0.0
module swap PrgEnv-cray PrgEnv-gnu
module load cray-python/3.8.5.0
module load cray-mpich/7.7.16
module load Boost/1.70.0-CrayGNU-20.11-python3
module load cudatoolkit/11.2.0_3.39-2.1__gf93aa1c
module load graphviz/2.44.0
set -e


# Setup RDMA for GPU. Set PIPE size to 256 (# of messages allowed in flight)
# Turn as-soon-as-possible transfer (NEMESIS_ASYNC_PROGRESS) on
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_G2G_PIPELINE=256
export MPICH_NEMESIS_ASYNC_PROGRESS=1
export MPICH_MAX_THREAD_SAFETY=multiple

# the eve toolchain requires a clang-format executable, we point it to the right place
export CLANG_FORMAT_EXECUTABLE=/project/s1053/install/venv/vcm_1.0/bin/clang-format

# run tests
echo "restoring cache"

FV3CORE_DIR=$SCRIPT_DIR/..
GT4PY_VERSION=$(cat $FV3CORE_DIR/GT4PY_VERSION.txt)
VENV_PATH=./venv

cache_key=v1-fv3core-$($SCRIPT_DIR/checksum.sh $SCRIPT_DIR/test_regression.sh $SCRIPT_DIR/run_test_cmd.sh $FV3CORE_DIR/constraints.txt)-$target

$SCRIPT_DIR/cache.sh restore $cache_key

echo "running tests"

python3 -m venv $VENV_PATH
. $VENV_PATH/bin/activate
python3 -m pip install --upgrade pip wheel

python3 -m pip install -c $FV3CORE_DIR/constraints.txt -e $FV3CORE_DIR/external/fv3gfs-util/
python3 -m pip install -c $FV3CORE_DIR/constraints.txt -r $FV3CORE_DIR/requirements/requirements_base.txt
python3 -m pip install -c $FV3CORE_DIR/constraints.txt git+https://github.com/ai2cm/gt4py.git@$GT4PY_VERSION#egg=gt4py$gt4py_extras
python3 -m pip install -c $FV3CORE_DIR/constraints.txt -e $FV3CORE_DIR

if [ "${target}" == "gpu" ] ; then
    set +e
    module unload cray-python
    module unload pycuda
    set -e
fi

if [ "`hostname | grep daint`" != "" ] ; then
    export PYTHONPATH=/project/s1053/install/serialbox/gnu/python:$PYTHONPATH
    run_on_daint_slurm "$command" $job_name
else
    python3 -m pip install -c $FV3CORE_DIR/constraints.txt -e $FV3CORE_DIR/external/serialbox/src/serialbox-python
    bash -c "$command"
fi

echo "saving cache"

$SCRIPT_DIR/cache.sh save $cache_key $VENV_PATH

deactivate

rm -rf $VENV_PATH

exit 0
