#!/usr/bin/env bash

backend=$1
expname=$2
cache_dir=${3:-/scratch/snx3000/olifu/jenkins/scratch/gt_caches_v2/$expname/$backend}

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
pace_dir=$script_dir/../

export gt4py_version=$(git submodule status $pace_dir/external/gt4py | awk '{print $1;}')

if ! compgen -G ./.gt_cache* > /dev/null; then
    if [ -d $cache_dir ]; then
        cache_filename=$cache_dir/$gt4py_version.tar.gz
        if [ -f "$cache_filename" ]; then
            tar -xzf $cache_filename -C .
            echo "Caches sucessfully loaded from $cache_filename"
        else
            echo "Caches $cache_filename not found"
            exit 1
        fi
    else
        echo "Cache directory $cache_dir not found"
    fi
else
    echo "WARNING: $(pwd)/.gt_cache* already exists. Will not overwrite directory with caches."
    exit 1
fi
