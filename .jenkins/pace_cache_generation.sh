#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$SCRIPT_DIR/../
[[ "${NODE_NAME}" == *"daint"* ]] && source ~/.bashrc

set -e
export LONG_EXECUTION=1

.jenkins/jenkins.sh initialize_driver $backend $experiment

SANITIZED_BACKEND=$(echo $backend | sed 's/:/_/g') #sanitize the backend from any ':'
TARGET_DIR="/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$experiment/$SANITIZED_BACKEND/"

export GT4PY_VERSION=$(git submodule status $PACE_DIR/external/gt4py | awk '{print $1;}')
echo "GT4PY_VERSION is $GT4PY_VERSION"

[ -n "$experiment" ] || exitError 1001 $LINENO "experiment is not defined"
[ -n "$SANITIZED_BACKEND" ] || exitError 1002 $LINENO "backend is not defined"

echo "Pruning cache to make sure no __pycache__ and *_pyext_BUILD dirs are present"
find .gt_cache* -type d -name \*_pyext_BUILD -prune -exec \rm -rf {} \;
find .gt_cache* -type d -name __pycache__ -prune -exec \rm -rf {} \;

CACHE_ARCHVE=$TARGET_DIR/$GT4PY_VERSION.tar.gz

echo "Copying gt4py cache directories to $CACHE_DIR"
tar -czf _tmp .gt_cache*
rm -rf $CACHE_ARCHVE

mkdir -p $CACHE_DIR
cp _tmp $CACHE_ARCHVE
