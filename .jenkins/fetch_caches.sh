#!/bin/bash
BACKEND=$1
EXPNAME=$2
SANITIZED_BACKEND=`echo $BACKEND | sed 's/:/_/g'` #sanitize the backend from any ':'
CACHE_DIR="/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/${EXPNAME}/${SANITIZED_BACKEND}"
REMOTE_CACHE_DIR="gs://fv3core-gt-cache/pace/${experiment}/${SANITIZED_BACKEND}/"
REMOTE_CACHE_FILENAME=${REMOTE_CACHE_DIR}/${GT4PY_VERSION}.tar.gz
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$SCRIPT_DIR/../

if [ -z "${GT4PY_VERSION}" ]; then
    export GT4PY_VERSION=`git submodule status ${PACE_DIR}/external/gt4py | awk '{print $1;}'`
fi

if [ ! -d $(pwd)/.gt_cache ]; then
    if [ -d ${CACHE_DIR} ]; then
        cache_filename=${CACHE_DIR}/${GT4PY_VERSION}.tar.gz
        if [ ! -f "${cache_filename}" ]; then
            cache_filename=${GT4PY_VERSION}.tar.gz
            gsutil cp $REMOTE_CACHE_FILENAME cache_filename
        fi
        tar -xf ${cache_filename} -C .
        echo ".gt_cache successfully fetched from ${cache_filename}"
    fi
fi
