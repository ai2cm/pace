#!/usr/bin/env bash

set -e

cmd=$1
key=$2
paths=( "$@" )
unset paths[0]  # cmd is not a path to be cached
unset paths[1]  # key is not a path to be cached

if [ "$cmd" != "save" ] && [ "$cmd" != "restore" ]; then
    echo "cmd must be one of 'save' or 'restore', got $cmd"
    exit 1
fi

if [ -z $PACE_CACHE_DIR ]; then
    cache_dir=~/.cache/pace
    mkdir -p $cache_dir
else
    cache_dir=$PACE_CACHE_DIR
fi

key_cache=$cache_dir/$key
target_dir=$(pwd)


if [ "$cmd" == "save" ]; then
    if [ -d $key_cache ]; then
        echo "cache for key $key_cache already exists, skipping cache step"
    else
        mkdir -p $key_cache
        for path in "${paths[@]}"; do
            cp -r $path $key_cache/$path
        done
        echo "cache stored for key $key"
    fi
elif [ "$cmd" == "restore" ]; then
    if [ -d $key_cache ]; then
        cd $key_cache
        for path in $(find . -maxdepth 1 -mindepth 1 -type d) ; do
            echo $path
            mkdir -p $target_dir/$path
            cp -r $path/* $target_dir/$path/
        done
        echo "cache restored for key $key"
    else
        echo "cache for key $key does not exist, skipping cache restoration"
    fi
fi
