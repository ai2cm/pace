#!/bin/bash
BACKEND=$1
EXPNAME=$2
SANITIZED_BACKEND=`echo $BACKEND | sed 's/:/_/g'` #sanitize the backend from any ':'
CACHE_DIR="/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches"
if [ ! -d $(pwd)/fv3gfs-physics/.gt_cache ]; then
    if [ -d ${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND} ]; then
        version_file=${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND}/GT4PY_VERSION.txt
        if [ -f ${version_file} ]; then
            version=`cat ${version_file}`
        else
            version=""
        fi
        if [ "$version" == "$GT4PY_VERSION" ]; then
            if [ -d ${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND}/.gt_cache  ]; then
                cp -r ${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND}/.gt_cache fv3gfs-physics/.
                find fv3gfs-physics/. -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/pace-physics-cache-setup\/backend\/${SANITIZED_BACKEND}\/experiment\/${EXPNAME}\/slave\/daint_submit/fv3gfs-physics|$(pwd)/fv3gfs-physics|g" {} +
                echo "gt_cache successfully fetched from: ${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND}/.gt_cache"
            else
                echo "gt_cache not fetched, caches not found in: ${CACHE_DIR}/${EXPNAME}/${SANITIZED_BACKEND}/.gt_cache"
            fi
        fi
    fi
fi
