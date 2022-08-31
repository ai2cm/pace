#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ "${target}" == "gpu" ] ; then
    # we only run this on HPC
    set +e
    module load cray-python
    module load pycuda
    set -e
fi

# run tests
echo "restoring cache"

UTIL_DIR=$SCRIPT_DIR/..

cache_key=v1-util-$($SCRIPT_DIR/checksum.sh $SCRIPT_DIR/test_util.sh $UTIL_DIR/requirements.txt $UTIL_DIR/requirements_gpu.txt $UTIL_DIR/../constraints.txt)-$target

$SCRIPT_DIR/cache.sh restore $cache_key

echo "running tests"

python3 -m venv venv
. ./venv/bin/activate

if [ "${target}" == "gpu" ] ; then
    set +e
    module unload cray-python
    module unload pycuda
    set -e
    pip3 install -r $UTIL_DIR/requirements.txt -r $UTIL_DIR/requirements_gpu.txt -c $UTIL_DIR/../constraints.txt -e $UTIL_DIR
else
    pip3 install -r $UTIL_DIR/requirements.txt -c $UTIL_DIR/../constraints.txt -e $UTIL_DIR
fi

pytest --junitxml results.xml $UTIL_DIR/tests

echo "saving cache"

$SCRIPT_DIR/cache.sh save $cache_key venv

deactivate

exit 0
