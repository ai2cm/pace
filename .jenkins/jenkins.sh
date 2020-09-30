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
optarg="$2"
optarg2="$3"
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

# get root directory of where jenkins.sh is sitting
root=`dirname $0`

# load machine dependent environment
if [ ! -f ${envloc}/env/env.${host}.sh ] ; then
    exitError 1202 ${LINENO} "could not find ${envloc}/env/env.${host}.sh"
fi
. ${envloc}/env/env.${host}.sh

# check if action script exists
script="${root}/actions/${action}.sh"
test -f "${script}" || exitError 1301 ${LINENO} "cannot find script ${script}"

# load scheduler tools
. ${envloc}/env/schedulerTools.sh
scheduler_script="`dirname $0`/env/submit.${host}.${scheduler}"

# if there is a scheduler script, make a copy for this job
if [ -f ${scheduler_script} ] ; then
    cp  ${scheduler_script} job_${action}.sh
    scheduler_script=job_${action}.sh
fi

# if this is a parallel job and the number of ranks is specified in optarg2, set NUM_RANKS
# and update the scheduler script if there is one
if grep -q "parallel" <<< "${script}"; then
    if grep -q "ranks" <<< "${optarg2}"; then
	export NUM_RANKS=`echo ${optarg2} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`
	echo "Setting NUM_RANKS=${NUM_RANKS}"
	if [ -f ${scheduler_script} ] ; then
	    sed -i 's|<NTASKS>|<NTASKS>\n#SBATCH \-\-hint=multithread\n#SBATCH --ntasks-per-core=2|g' ${scheduler_script}
	    sed -i 's|45|30|g' ${scheduler_script}
	    sed -i 's|cscsci|debug|g' ${scheduler_script}
	    sed -i 's|<NTASKS>|"'${NUM_RANKS}'"|g' ${scheduler_script}
	    sed -i 's|<NTASKSPERNODE>|"24"|g' ${scheduler_script}
	fi
    fi
fi

module load daint-gpu
module add "${installdir}/modulefiles/"
module load gcloud
echo "UPSTREAM_PROJECT: ${UPSTREAM_PROJECT}"
echo "UPSTREAM_BUILD_NUMBER: ${UPSTREAM_BUILD_NUMBER}"
if [ ! -z "${UPSTREAM_PROJECT}" ] ; then
    # Set in build_for_daint jenkins plan, to mark what fv3core image to pull
    export FV3_TAG="${UPSTREAM_PROJECT}-${UPSTREAM_BUILD_NUMBER}"
    echo "Downstream project using FV3_TAG=${FV3_TAG}"
fi
# If using sarus, load the image and set variables for running tests,
# otherwise build the image
if [ ${container_engine} == "sarus" ]; then
    module load sarus
    export FV3_IMAGE="load/library/${FV3_TAG}"
    echo "Using FV3_IMAGE=${FV3_IMAGE}"
    make sarus_load_tar
    if grep -q "parallel" <<< "${script}"; then
	export CONTAINER_ENGINE="srun sarus"
	export RUN_FLAGS="--mpi"
	export MPIRUN_CALL=""
    else
	export CONTAINER_ENGINE="sarus"
	export RUN_FLAGS=""
    fi
else
    make build
fi

# get the test data version from the Makefile
export FORTRAN_VERSION=`grep "FORTRAN_SERIALIZED_DATA_VERSION=" Makefile  | cut -d '=' -f 2`

# Set the SCRATCH directory to the working directory if not set (e.g. for running on gce)
if [ -z ${SCRATCH} ] ; then
    export SCRATCH=`pwd`
fi

# Set the host data location
export TEST_DATA_DIR="${SCRATCH}/fv3core_fortran_data/${FORTRAN_VERSION}"

G2G="false"

# Run the jenkins command
run_command "${script} ${optarg} ${optarg2} " Job${action} ${G2G} ${scheduler_script}

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
