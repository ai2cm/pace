#!/bin/bash

# Jenkins action to run baroclinic initialization at scale on Piz Daint

# Syntax:
# .jenkins/action/baroclinic_initialization.sh <option>

## Arguments:

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
experiment="$1"

ARTIFACT_ROOT="/project/s1053/baroclinic_initialization/"

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILDENV_DIR=$JENKINS_DIR/../buildenv
PACE_DIR=$JENKINS_DIR/../

# load machine dependent environment
if [ ! -f ${BUILDENV_DIR}/env.${host}.sh ] ; then
    exitError 1202 ${LINENO} "could not find ${BUILDENV_DIR}/env.${host}.sh"
fi
. ${BUILDENV_DIR}/env.${host}.sh

# load scheduler tools
. ${BUILDENV_DIR}/schedulerTools.sh
scheduler_script="${BUILDENV_DIR}/submit.${host}.${scheduler}"

# if there is a scheduler script, make a copy for this job
if [ -f ${scheduler_script} ] ; then
    if [ "${action}" == "setup" ]; then
	scheduler="none"
    else
	cp  ${scheduler_script} job_initialization.sh
	scheduler_script=job_initialization.sh
    fi
fi

if grep -q "ranks" <<< "${experiment}"; then
    export NUM_RANKS=`echo ${experiment} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`
    echo "Setting NUM_RANKS=${NUM_RANKS}"
    if [ -f ${scheduler_script} ] ; then
        sed -i 's|<NTASKS>|<NTASKS>\n#SBATCH \-\-hint=multithread\n#SBATCH --ntasks-per-core=2|g' ${scheduler_script}
        sed -i 's|45|30|g' ${scheduler_script}
        if [ "$NUM_RANKS" -gt "6" ] && [ ! -v LONG_EXECUTION ]; then
            sed -i 's|cscsci|debug|g' ${scheduler_script}
        elif [ "$NUM_RANKS" -gt "6" ]; then
            sed -i 's|cscsci|normal|g' ${scheduler_script}
        fi
        sed -i 's|<NTASKS>|"'${NUM_RANKS}'"|g' ${scheduler_script}
        sed -i 's|<NTASKSPERNODE>|"24"|g' ${scheduler_script}
    fi
fi

if [ ${python_env} == "virtualenv" ]; then
    if [ -d ${VIRTUALENV} ]; then
	echo "Using existing virtualenv ${VIRTUALENV}"
    else
	echo "virtualenv ${VIRTUALENV} is not setup yet, installing now"
	export PACE_INSTALL_FLAGS="-e"
	${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
    fi
    source ${VIRTUALENV}/bin/activate
    if grep -q "parallel" <<< "${script}"; then
	export MPIRUN_CALL="srun"
    fi
    export PACE_PATH="${JENKINS_DIR}/../"
    export TEST_DATA_RUN_LOC=${TEST_DATA_HOST}
fi

CMD="${MPIRUN_CALL} python3 ${PACE_DIR}/driver/examples/baroclinic_init.py ${JENKINS_DIR}/driver_configs/${experiment}.yaml"
run_command "${CMD}" Job${action} ${scheduler_script}
pip install matplotlib
python3 ${PACE_DIR}/driver/examples/plot_baroclinic_init.py ${PACE_DIR}/output.zarr
mkdir -p ${ARTIFACT_ROOT}/${experiment}
mv *.png ${ARTIFACT_ROOT}/${experiment}/.
