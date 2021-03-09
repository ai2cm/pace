#!/bin/bash

set -e -x

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"
TESTDATA_PATH="/project/s1053/fv3core_serialized_test_data"
FORTRAN_SERIALIZED_DATA_VERSION=7.2.5
test -n "${experiment}" || exitError 1001 ${LINENO} "experiment is not defined"
test -n "${backend}" || exitError 1002 ${LINENO} "backend is not defined"

data_path=${TESTDATA_PATH}/${FORTRAN_SERIALIZED_DATA_VERSION}/${experiment}

$ROOT_DIR/examples/standalone/benchmarks/run_on_daint.sh 2 6 $backend /project/s1053/performance/fv3core_monitor/$backend/ $data_path

rm -rf .gt_cache_0000*
