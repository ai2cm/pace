#!/usr/bin/env bash

set -e -x

cmd=$1
key=$2
paths=( "$@" )
unset paths[0]  # cmd is not a path to be cached
unset paths[1]  # key is not a path to be cached

if [ "$cmd" == "save" ] && [ "${#paths[@]}" == 0 ]; then
    echo "when cmd is 'save', at least one path must be given to save"
    exit 1
fi

if [ "$cmd" != "save" ] && [ "$cmd" != "restore" ]; then
    echo "cmd must be one of 'save' or 'restore', got $cmd"
    exit 1
fi

if [ -z $PACE_CACHE_DIR ]; then
    cache_dir=~/.cache/pace
else
    cache_dir=$PACE_CACHE_DIR
fi
mkdir -p $cache_dir

key_cache=$cache_dir/$key.tar.gz
target_dir=$(pwd)


if [ "$cmd" == "save" ]; then
    if [ -f "$key_cache" ]; then
        echo "cache for key $key_cache already exists, skipping cache step"
    else
        tar -czf $key_cache ${paths[*]}
        echo "cache stored for key $key"
    fi
elif [ "$cmd" == "restore" ]; then
    if [ -f "$key_cache" ]; then
        tar -xf $key_cache -C $target_dir/
        echo "cache restored for key $key"
    else
        echo "cache for key $key does not exist, skipping cache restoration"
    fi
fi
