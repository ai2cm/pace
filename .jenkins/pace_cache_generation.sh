#!/usr/bin/env bash

set -e

if (( $# < 3 )); then
    echo "USAGE: .jenkins/pace_cache_generation.sh backend experiment [target_dir] [bypass_wrapper]"
fi

backend=$1
experiment=$2
shift 2

if [ $# > 0 ]; then
    target_dir=$1
else
    target_dir="/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$experiment/${backend//:/_}"
fi

if [ $# > 1 ] && [ $2 == "bypass_wrapper" ]; then
    bypass_wrapper=true
else
    bypass_wrapper=false
fi

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pace_dir=$script_dir/../
[[ "${NODE_NAME}" == *"daint"* ]] && source ~/.bashrc

if [[ $bypass_wrapper != "true" ]]; then
    export LONG_EXECUTION=1
    slave=daint .jenkins/jenkins.sh create_caches $backend $experiment
else
    .jenkins/actions/create_caches.sh $backend $experiment
fi

export gt4py_version=$(git submodule status $pace_dir/external/gt4py | awk '{print $1;}')
echo "Using gt4py version $gt4py_version"

echo "Pruning cache to make sure no __pycache__ and *_pyext_BUILD dirs are present"
find .gt_cache* -type d -name \*_pyext_BUILD -prune -exec \rm -rf {} \;
find .gt_cache* -type d -name __pycache__ -prune -exec \rm -rf {} \;

cache_archive=$target_dir/$gt4py_version.tar.gz

echo "Copying gt4py cache directories to $cache_archive"
tar -czf _tmp .gt_cache*
rm -rf $cache_archive

mkdir -p $target_dir
cp _tmp $cache_archve
