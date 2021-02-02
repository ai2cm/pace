#!/bin/bash
# This is the master script used to trigger Jenkins actions.
# The idea of this script is to keep the amount of code in the "Execute shell" field small
#
# Example syntax:
# .jenkins/jenkins.sh run_regression_tests
#
# Other actions such as test/build/deploy can be defined.

### Some environment variables available from Jenkins
### Note: for a complete list see https://jenkins.ginko.ch/env-vars.html
# slave              The name of the build host (daint, kesch, ...).
# BUILD_NUMBER       The current build number, such as "153".
# BUILD_ID           The current build id, such as "2005-08-22_23-59-59" (YYYY-MM-DD_hh-mm-ss).
# BUILD_DISPLAY_NAME The display name of the current build, something like "#153" by default.
# NODE_NAME          Name of the host.
# NODE_LABELS        Whitespace-separated list of labels that the node is assigned.
# JENKINS_HOME       The absolute path of the data storage directory assigned on the master node.
# JENKINS_URL        Full URL of Jenkins, like http://server:port/jenkins/
# BUILD_URL          Full URL of this build, like http://server:port/jenkins/job/foo/15/
# JOB_URL            Full URL of this job, like http://server:port/jenkins/job/foo/

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

# echo basic setup
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"

# start timer
T="$(date +%s)"

# check sanity of environment
test -n "$1" || exitError 1001 ${LINENO} "must pass an argument"
test -n "${slave}" || exitError 1005 ${LINENO} "slave is not defined"

# some global variables
action="$1"
backend="$2"
experiment="$3"
# check presence of env directory
pushd `dirname $0` > /dev/null
envloc=`/bin/pwd`
popd > /dev/null
shopt -s expand_aliases
# Download the env
. ${envloc}/env.sh

# setup module environment and default queue
test -f ${envloc}/env/machineEnvironment.sh || exitError 1201 ${LINENO} "cannot find machineEnvironment.sh script"
. ${envloc}/env/machineEnvironment.sh
export python_env=${python_env}
echo "PYTHON env ${python_env}"
# get root directory of where jenkins.sh is sitting
export jenkins_dir=`dirname $0`

# load machine dependent environment
if [ ! -f ${envloc}/env/env.${host}.sh ] ; then
    exitError 1202 ${LINENO} "could not find ${envloc}/env/env.${host}.sh"
fi
. ${envloc}/env/env.${host}.sh

# check if action script exists
script="${jenkins_dir}/actions/${action}.sh"
test -f "${script}" || exitError 1301 ${LINENO} "cannot find script ${script}"

# load scheduler tools
. ${envloc}/env/schedulerTools.sh
scheduler_script="`dirname $0`/env/submit.${host}.${scheduler}"

# if there is a scheduler script, make a copy for this job
if [ -f ${scheduler_script} ] ; then
    if [ "${action}" == "setup" ]; then
	scheduler="none"
    else
	cp  ${scheduler_script} job_${action}.sh
	scheduler_script=job_${action}.sh
    fi
fi

# if this is a parallel job and the number of ranks is specified in the experiment argument, set NUM_RANKS
# and update the scheduler script if there is one
if grep -q "parallel" <<< "${script}"; then
    if grep -q "ranks" <<< "${experiment}"; then
	export NUM_RANKS=`echo ${experiment} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`
	echo "Setting NUM_RANKS=${NUM_RANKS}"
	if grep -q "cuda" <<< "${backend}" ; then
	    export MPICH_RDMA_ENABLED_CUDA=1
	else
	    export MPICH_RDMA_ENABLED_CUDA=0
	fi
	if [ -f ${scheduler_script} ] ; then
	    sed -i 's|<NTASKS>|<NTASKS>\n#SBATCH \-\-hint=multithread\n#SBATCH --ntasks-per-core=2|g' ${scheduler_script}
	    sed -i 's|45|30|g' ${scheduler_script}
	    if [ "$NUM_RANKS" -gt "6" ];then
            sed -i 's|cscsci|debug|g' ${scheduler_script}
        fi
	    sed -i 's|<NTASKS>|"'${NUM_RANKS}'"|g' ${scheduler_script}
	    sed -i 's|<NTASKSPERNODE>|"24"|g' ${scheduler_script}
	fi
    fi
fi

# set thresholds override file if it exists
test_type=${experiment##*_}
OVERRIDES_FOLDER="${envloc}/../tests/translate/overrides/"
OVERRIDES_FILE="${OVERRIDES_FOLDER}/${test_type}.yaml"
echo "overrides file:"
echo ${OVERRIDES_FILE}
if test -f "${OVERRIDES_FILE}"; then
    echo "OVERRIDE"
    export MOUNTS=" -v ${OVERRIDES_FOLDER}:/thresholds"
    if [ ${python_env} == "virtualenv" ]; then
	threshold_folder=${OVERRIDES_FOLDER}
    else
	threshold_folder="/thresholds"
    fi
    export THRESH_ARGS="--threshold_overrides_file=${threshold_folder}/${test_type}.yaml"
fi
export PROF_FOLDER="${envloc}/../prof"
`mkdir -p ${PROF_FOLDER}`
export MOUNTS="${MOUNTS} -v ${PROF_FOLDER}:/prof"
module load daint-gpu
module add "${installdir}/modulefiles/"
module load gcloud

# get the test data version from the Makefile
export FORTRAN_VERSION=`grep "FORTRAN_SERIALIZED_DATA_VERSION=" Makefile  | cut -d '=' -f 2`

# Set the SCRATCH directory to the working directory if not set (e.g. for running on gce)
if [ -z ${SCRATCH} ] ; then
    export SCRATCH=`pwd`
fi

# Set the host data head directory location
export TEST_DATA_DIR="${SCRATCH}/fv3core_fortran_data/${FORTRAN_VERSION}"
export FV3_STENCIL_REBUILD_FLAG=False
# Set the host data location
export TEST_DATA_HOST="${TEST_DATA_DIR}/${experiment}/"
export EXPERIMENT=${experiment}
if [ -z ${JENKINS_TAG} ]; then
    export JENKINS_TAG=${JOB_NAME}-${BUILD_NUMBER}
fi
echo "JENKINS TAG ${JENKINS_TAG}"
if [ -z ${VIRTUALENV} ]; then
    echo "setting VIRTUALENV"
    export VIRTUALENV=${WORKSPACE}/vcm_env_${JENKINS_TAG}
fi

if [ ${python_env} == "virtualenv" ]; then
    if [ -d ${VIRTUALENV} ]; then
	echo "Using existing virtualenv ${VIRTUALENV}"
	source ${VIRTUALENV}/bin/activate
    else
	echo "virtualenv is not setup yet"
    fi
    if grep -q "parallel" <<< "${script}"; then
	export MPIRUN_CALL="srun"
    fi
    export FV3_PATH="${envloc}/../"
    export TEST_DATA_RUN_LOC=${TEST_DATA_HOST}
    export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:$PYTHONPATH
fi

G2G="false"
export DOCKER_BUILDKIT=1

run_command "${script} ${backend} ${experiment} " Job${action} ${G2G} ${scheduler_script}

if [ $? -ne 0 ] ; then
  exitError 1510 ${LINENO} "problem while executing script ${script}"
fi
echo "### ACTION ${action} SUCCESSFUL"

# end timer and report time taken
T="$(($(date +%s)-T))"
printf "####### time taken: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0

# so long, Earthling!
