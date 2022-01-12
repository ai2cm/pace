#!/bin/bash
set -e -x
BACKEND=$1
SANITIZED_BACKEND=`echo $BACKEND | sed 's/:/_/g'` #sanitize the backend from any ':'
EXPNAME=$2
export TEST_ARGS="-v -s -rsx --backend=${BACKEND} --which_modules=FVDynamics"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# sync the test data
make get_test_data

$SCRIPT_DIR/fetch_caches.sh $BACKEND $EXPNAME

CONTAINER_CMD="" make savepoint_tests_mpi
