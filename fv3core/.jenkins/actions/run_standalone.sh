#!/bin/bash

# Jenkins action to run a benchmark of dynamics.py on Piz Daint
# 3/11/2021, Tobias Wicky, Vulcan Inc

# Syntax:
# .jenkins/action/run_standlone.sh <option>

## Arguments:
# $1: <option> which can be either empty, "profile" or "build_cache"

# stop on all errors and echo commands
set -e

# utility function for error handling
exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

# check arguments
DO_PROFILE="false"
DO_NSYS_RUN="false"
SAVE_CACHE="false"
SAVE_TIMINGS="false"
SAVE_ARTIFACTS="true"

if [ "$1" == "profile" ] ; then
    DO_PROFILE="true"
fi
# Extra run in 'cuda' with nsys
if [ "${DO_PROFILE}" == "true" ] && [ "${backend}" == "cuda" ] ; then
    DO_NSYS_RUN="true"
fi
if [ "$1" == "build_cache" ] ; then
    SAVE_CACHE="true"
fi
# only save timings if this is neither a cache build nor a profiling run
if [ "${SAVE_CACHE}" != "true" -a "${DO_PROFILE}" != "true" ] ; then
    SAVE_TIMINGS="true"
fi
# check if we store the results of this run
if [[ "$GIT_BRANCH" != "origin/main" ]]; then
    SAVE_ARTIFACTS="false"
fi

# configuration
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"
DATA_VERSION=`grep 'FORTRAN_SERIALIZED_DATA_VERSION *=' ${ROOT_DIR}/Makefile | cut -d '=' -f 2`
if [ "$2" != "" ]; then
    TIMESTEPS=$2
elif [ "${SAVE_CACHE}" == "true" ]; then
    TIMESTEPS=2
else
    TIMESTEPS=60
fi

# Could parse from namelist, ranks = 6 * layout[0] * layout[1]
RANKS=`echo ${experiment} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`
BENCHMARK_DIR=${ROOT_DIR}/examples/standalone/benchmarks
DATA_DIR="/project/s1053/fv3core_serialized_test_data/${DATA_VERSION}/${experiment}/driver"
ARTIFACT_ROOT="/project/s1053/performance/"
TIMING_DIR="${ARTIFACT_ROOT}/fv3core_performance/${backend}"
PROFILE_DIR="${ARTIFACT_ROOT}/fv3core_profile/${backend}"


# check sanity of environment
test -n "${experiment}" || exitError 1001 ${LINENO} "experiment is not defined"
test -n "${backend}" || exitError 1002 ${LINENO} "backend is not defined"
if [ ! -d "${DATA_DIR}" ] ; then
    exitError 1003 ${LINENO} "test data directory ${DATA_DIR} does not exist"
fi
if [ ! -d "${ARTIFACT_ROOT}" ] ; then
    exitError 1004 ${LINENO} "Artifact directory ${ARTIFACT_ROOT} does not exist"
fi
if [ ! -d "${BENCHMARK_DIR}" ] ; then
    exitError 1005 ${LINENO} "Benchmark directory ${BENCHMARK_DIR} does not exist"
fi

# GTC backend name fix: passed as gtc_gt_* but their real name are gtc:gt:*
#                       OR gtc_* but their real name is gtc:*
if [[ $backend = gtc_gt_* ]] ; then
    # sed explained: replace _ with :, two times
    backend=`echo $backend | sed 's/_/:/;s/_/:/'`
fi
if [[ $backend = gtc_* ]] ; then
    # sed explained: replace _ with :
    backend=`echo $backend | sed 's/_/:/'`
fi

# echo config
echo "=== $0 configuration ==========================="
echo "Script:                       ${SCRIPT}"
echo "Do profiling:                 ${DO_PROFILE}"
echo "Save GT4Py caches:            ${SAVE_CACHE}"
echo "Save timings:                 ${SAVE_TIMINGS}"
echo "Save Aritfacts:               ${SAVE_ARTIFACTS}"
echo "Root directory:               ${ROOT_DIR}"
echo "Experiment:                   ${experiment}"
echo "Backend:                      ${backend}"
echo "Data version:                 ${DATA_VERSION}"
echo "Timesteps:                    ${TIMESTEPS}"
echo "Ranks:                        ${RANKS}"
echo "Benchmark directory:          ${BENCHMARK_DIR}"
echo "Data directory:               ${DATA_DIR}"
echo "Perf. artifact directory:     ${TIMING_DIR}"
echo "Profile artifact directory:   ${PROFILE_DIR}"


# If the backend is a GTC backend we fetch the caches
if [[ $backend != *numpy* ]];then
    . ${ROOT_DIR}/../.jenkins/fetch_caches.sh $backend $experiment
fi

# run standalone
echo "=== Running standalone ========================="
if [ "${DO_PROFILE}" == "true" ] ; then
    profile="--profile"
fi
cmd="${BENCHMARK_DIR}/run_on_daint.sh ${TIMESTEPS} ${RANKS} ${backend} ${DATA_DIR}"
echo "Run command: ${cmd} \"\" \"${profile}\""
${cmd} "" "${profile}" "${DO_NSYS_RUN}"
echo "=== Post-processing ============================"

# store timing artifacts
if [ "${SAVE_TIMINGS}" == "true" ] && [ "${SAVE_ARTIFACTS}" == "true" ] ; then
        echo "Copying timing information to ${TIMING_DIR}"
        cp $ROOT_DIR/*.json ${TIMING_DIR}/
fi

# copying the cache is in a separate action (generache_cache.sh),
# otherwise delete it
if [ "${SAVE_CACHE}" != "true" ] ; then
    rm -rf .gt_cache*
fi

# store profiling artifacts
if [ "${DO_PROFILE}" == "true" ] ; then
    echo "Analyzing profiling results"
    if [ "${SAVE_ARTIFACTS}" == "true" ] ; then
        echo "Copying profiling information to ${PROFILE_DIR}/prof/"
        rm -f ${PROFILE_DIR}/prof/*.prof || true
        cp $ROOT_DIR/*.prof ${PROFILE_DIR}/prof/
    fi
fi

# copy nsys results - cleaning the old ones first
if [ "${DO_NSYS_RUN}" == "true" ] ; then
    echo "Copying new nsys results to ${PROFILE_DIR}/nsys/"
    rm -f ${PROFILE_DIR}/nsys/*.qdrep || true
    cp $ROOT_DIR/*.qdrep ${PROFILE_DIR}/nsys/
fi

# remove venv (too many files!)
rm -rf $ROOT_DIR/venv

echo "=== Done ======================================="
