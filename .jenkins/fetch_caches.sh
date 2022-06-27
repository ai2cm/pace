#!/bin/bash
BACKEND=$1
EXPNAME=$2
CACHE_TYPE=$3
SANITIZED_BACKEND=`echo $BACKEND | sed 's/:/_/g'` #sanitize the backend from any ':'
CACHE_DIR="/scratch/snx3000/olifu/jenkins/scratch/gt_caches_v1/${CACHE_TYPE}/${EXPNAME}/${SANITIZED_BACKEND}"
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$SCRIPT_DIR/../

if [ -z "${GT4PY_VERSION}" ]; then
    export GT4PY_VERSION=`git submodule status ${PACE_DIR}/external/gt4py | awk '{print $1;}'`
fi

if ! compgen -G ./.gt_cache* > /dev/null; then
    if [ -d ${CACHE_DIR} ]; then
        cache_filename=${CACHE_DIR}/${GT4PY_VERSION}.tar.gz
        if [ -f "${cache_filename}" ]; then
            tar -xzf ${cache_filename} -C .
            echo ".gt_cache successfully fetched from ${cache_filename}"
        else
            echo ".gt_cache not fetched, cache not found at ${cache_filename}"
        fi
    fi
else
    echo "WARNING: $(pwd)/.gt_cache already exists. Will not overwrite directory with caches."
    echo "Please remove this directory and try again."
fi
