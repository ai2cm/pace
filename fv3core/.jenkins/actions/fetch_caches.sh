#!/bin/bash
BACKEND=$1
EXPNAME=$2
SANITIZED_BACKEND=`echo $BACKEND | sed 's/:/_/g'` #sanitize the backend from any ':'
CACHE_DIR="/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/"
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$SCRIPT_DIR/../../../

if [ -z "${GT4PY_VERSION}" ]; then
    export GT4PY_VERSION=`git submodule status ${PACE_DIR}/external/gt4py | awk '{print $1;}'`
fi

if [ ! -d $(pwd)/.gt_cache ]; then
    if [ -d ${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND} ]; then
        version_file=${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND}/GT4PY_VERSION.txt
        if [ -f ${version_file} ]; then
            version=`cat ${version_file}`
        else
            version=""
        fi
        if [ "$version" == "$GT4PY_VERSION" ]; then
            if [ -d ${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND}/.gt_cache  ]; then
                cp -r ${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND}/.gt_cache .
                find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/pace-fv3core-cache-setup\/backend\/${SANITIZED_BACKEND}\/experiment\/${EXPNAME}\/slave\/daint_submit/fv3core|$(pwd)|g" {} +
            fi
        fi
    fi
fi
