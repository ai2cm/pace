#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} --which_modules=FVDynamics"

# sync the test data
make get_test_data

cp -r /scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$EXPNAME/.gt_cache_0000* .
find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/zz_fv3core_cacheSetup\/backend\/gtx86\/experiment\/$EXPNAME\/slave\/daint_submit|$(pwd)|g" {} +

make tests_venv_mpi
