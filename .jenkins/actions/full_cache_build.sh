#!/bin/bash
set -e -x
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"
TESTDATA_PATH="/scratch/snx3000/olifu/jenkins/scratch/fv3core_fortran_data/7.2.5/"
test -n "${experiment}" || exitError 1001 ${LINENO} "experiment is not defined"
test -n "${backend}" || exitError 1002 ${LINENO} "backend is not defined"


if [ "$experiment" = "c12_6ranks_standard" ]; then
    data_path="${TESTDATA_PATH}/c12_6ranks_standard"
fi
if [ "$experiment" = "c96_6ranks_baroclinic" ]; then
    data_path="${TESTDATA_PATH}/c96_6ranks_baroclinic"
fi
if [ "$experiment" = "c128_6ranks_baroclinic" ]; then
    data_path="${TESTDATA_PATH}/c128_6ranks_baroclinic"
fi

$ROOT_DIR/examples/standalone/benchmarks/run_on_daint.sh 1 6 $backend . $data_path
mkdir -p /scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$experiment/$backend
rm -rf /scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$experiment/$backend/.gt_cache_00000*
mv .gt_cache_00000* /scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$experiment/$backend/
