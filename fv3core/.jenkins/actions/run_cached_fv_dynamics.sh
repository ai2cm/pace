#!/bin/bash
set -e -x
BACKEND=$1
SANITIZED_BACKEND=`echo $BACKEND | sed 's/:/_/g'` #sanitize the backend from any ':'
EXPNAME=$2
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} --which_modules=FVDynamics"

# sync the test data
make get_test_data

if [ ! -d $(pwd)/.gt_cache ]; then
    version_file=/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/$SANITIZED_BACKEND/GT4PY_VERSION.txt
    if [ -f ${version_file} ]; then
	version=`cat ${version_file}`
    else
	version=""
    fi
    if [ "$version" == "$GT4PY_VERSION" ]; then
        cp -r /scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/$SANITIZED_BACKEND/.gt_cache* .
        find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/gtc_cache_setup\/backend\/${SANITIZED_BACKEND}\/experiment\/${EXPNAME}\/slave\/daint_submit/fv3core|$(pwd)|g" {} +
    fi
fi
CONTAINER_CMD="" make savepoint_tests_mpi
