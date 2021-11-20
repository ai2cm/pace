#!/usr/bin/env bash

# Read arguments
make_target="$1"
backend="$input_backend"
experiment="$3"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} "
export TEST_DATA_RUN_LOC=$SCRIPT_DIR/../test_data/$experiment
export CONTAINER_CMD=''

$SCRIPT_DIR/run_test_cmd.sh "make $make_target"
