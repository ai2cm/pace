#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} --which_modules=FVDynamics"

# sync the test data
make get_test_data

if [ ! -d $(pwd)/.gt_cache_000000 ]; then
    cp -r /scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/$BACKEND/.gt_cache_0000* .
    find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/fv3core-cache-setup\/backend\/$BACKEND\/experiment\/$EXPNAME\/slave\/daint_submit|$(pwd)|g" {} +
fi
make tests_venv_mpi
