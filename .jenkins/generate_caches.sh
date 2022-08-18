#!/bin/bash

# Script to generate gt caches on Piz Daint
# Syntax:
# .jenkins/generate_caches.sh <backend> <experiment> <cache_type>


# stop on all errors and echo commands
set -e -x

# utility function for error handling
exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

backend=$1
experiment=$2
cache_type=$3
SANITIZED_BACKEND=`echo $backend | sed 's/:/_/g'` #sanitize the backend from any ':'
CACHE_DIR="/scratch/snx3000/olifu/jenkins/scratch/gt_caches_v1/${cache_type}/${experiment}/${SANITIZED_BACKEND}/"
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$SCRIPT_DIR/../

if [ -z "${GT4PY_VERSION}" ]; then
    export GT4PY_VERSION=`git submodule status ${PACE_DIR}/external/gt4py | awk '{print $1;}'`
    echo "GT4PY_VERSION is ${GT4PY_VERSION}"
fi

CACHE_FILENAME=${CACHE_DIR}/${GT4PY_VERSION}.tar.gz

test -n "${experiment}" || exitError 1001 ${LINENO} "experiment is not defined"
test -n "${SANITIZED_BACKEND}" || exitError 1002 ${LINENO} "backend is not defined"

# store cache artifacts (and remove caches afterwards)
echo "Pruning cache to make sure no __pycache__ and *_pyext_BUILD dirs are present"
find .gt_cache* -type d -name \*_pyext_BUILD -prune -exec \rm -rf {} \;
find .gt_cache* -type d -name __pycache__ -prune -exec \rm -rf {} \;
echo "Copying GT4Py cache directories to ${CACHE_DIR}"
mkdir -p ${CACHE_DIR}
tar -czf _tmp .gt_cache*
mv _tmp ${CACHE_FILENAME}
