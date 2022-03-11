#!/bin/bash
set -e -x
BACKEND=$1
EXPNAME=$2
SAVEPOINT=${3:-All}
ORCHESTRATION=${4:-none}
XML_REPORT="sequential_test_results.xml"
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
JENKINS_DIR="$(dirname "$SCRIPT_DIR")"

if [ $SAVEPOINT == "All" ]; then
    export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "
else
    export TEST_ARGS="-v -s -rsx --backend=${BACKEND} --which_modules=${SAVEPOINT}"
fi

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../"

# sync the test data
make get_test_data

if [ ${python_env} == "virtualenv" ]; then
    export TEST_ARGS="${TEST_ARGS} --junitxml=${JENKINS_DIR}/${XML_REPORT}"
    CONTAINER_CMD="srun" make tests savepoint_tests
else
    export TEST_ARGS="${TEST_ARGS} --junitxml=/.jenkins/${XML_REPORT}"
    if [ ${ORCHESTRATION} != "none" ]; then
        VOLUMES="-v ${SCRIPT_DIR}/../:/.jenkins" DACE_INSTALL="/.jenkins/install_dace.sh ${ORCHESTRATION} &&" make savepoint_tests
    else
        VOLUMES="-v ${SCRIPT_DIR}/../:/.jenkins" make tests savepoint_tests
    fi
fi
