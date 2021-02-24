#!/bin/bash
set -e -x
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"
TESTDATA_PATH="/project/s1053/fv3core_serialized_test_data/"
test -n "${experiment}" || exitError 1001 ${LINENO} "experiment is not defined"
test -n "${backend}" || exitError 1002 ${LINENO} "backend is not defined"


if [ "$experiment" = "c12_6ranks_standard" ]; then
    data_path="${TESTDATA_PATH}7.0.0/c12_6ranks_standard"
fi
if [ "$experiment" = "c96_6ranks_baroclinic" ]; then
    data_path="${TESTDATA_PATH}7.2.3/c96_6ranks_baroclinic"
fi
if [ "$experiment" = "c128_6ranks_baroclinic" ]; then
    data_path="${TESTDATA_PATH}7.2.3/c128_6ranks_baroclinic"
fi

$ROOT_DIR/examples/standalone/benchmarks/run_on_daint.sh 2 6 $backend /project/s1053/performance/fv3core_monitor/$backend/ $data_path
rm -rf .gt_cache_0000*
