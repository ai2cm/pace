#!/usr/bin/env bash

set -e

if (( $# < 3 )); then
    echo "USAGE: .jenkins/pace_cache_generation.sh backend experiment [target_uri]"
fi

backend=$1
experiment=$2

# If the target_uri starts with "daint:" then the Jenkins scripts are used
# Use the target_uri as the last command line argument if passed, else the default daint path
default_target_uri="daint:/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$experiment/${backend//:/_}"
target_uri=${3:-$daint_target_uri}

if [[ $target_uri == daint:* ]]; then
    use_jenkins_action=true
    target_dir=${target_uri#daint:}
else
    use_jenkins_action=false
    target_dir=$target_uri
fi

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pace_dir=$script_dir/../
[[ "${NODE_NAME}" == *"daint"* ]] && source ~/.bashrc

if [[ $use_jenkins_action == "true" ]]; then
    export LONG_EXECUTION=1
    .jenkins/jenkins.sh initialize_driver $backend $experiment
else
    data_version=$(cd fv3core && EXPERIMENT=$experiment TARGET=driver make get_test_data | tail -1)
    $pace_dir/.jenkins/initialize_driver.py test_data/$data_version/$experiment/driver $backend
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
