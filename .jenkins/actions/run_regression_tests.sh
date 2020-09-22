#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
ARGS="-v -s -rsx --backend=${BACKEND} --junitxml=/.jenkins/sequential_test_results.xml"
export EXPERIMENT=${EXPNAME}

# Set the host data location
export TEST_DATA_HOST="${TEST_DATA_DIR}/${EXPNAME}/"

# sync the test data 
make get_test_data

make run_tests_sequential TEST_ARGS="${ARGS}"
