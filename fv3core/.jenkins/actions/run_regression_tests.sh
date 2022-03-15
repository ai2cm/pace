#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
XML_REPORT="sequential_test_results.xml"
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../"

# sync the test data
make get_test_data

if [ ${python_env} == "virtualenv" ]; then
    export TEST_ARGS="${TEST_ARGS} --junitxml=${JENKINS_DIR}/${XML_REPORT}"
    if [[ ${FV3_DACEMODE} == "True" ]]; then
        export DACE_execution_general_check_args=0
        export DACE_frontend_dont_fuse_callbacks=1
        export DACE_compiler_cpu_openmp_sections=0
        export DACE_compiler_cuda_max_concurrent_streams=-1
        export DACE_frontend_unroll_threshold=0
        export DACE_compiler_unique_functions=none
        CONTAINER_CMD="srun" make savepoint_tests
    else
        CONTAINER_CMD="srun" make tests savepoint_tests
    fi
else
    export TEST_ARGS="${TEST_ARGS} --junitxml=/.jenkins/${XML_REPORT}"
    if [[ ${FV3_DACEMODE} == "True" ]]; then
        VOLUMES="-v ${SCRIPT_DIR}/../:/.jenkins" RUN_FLAGS="--rm -e FV3_DACEMODE=True" make savepoint_tests
    else
        VOLUMES="-v ${SCRIPT_DIR}/../:/.jenkins" make tests savepoint_tests
    fi
fi
