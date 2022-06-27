#!/bin/bash

# Jenkins action to run baroclinic initialization at scale on Piz Daint

# Syntax:
# .jenkins/action/baroclinic_initialization.sh <option>

## Arguments:

# utility function for error handling
exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

# stop on all errors and echo commands
set -x +e

experiment="$1"
minutes=30

ARTIFACT_ROOT="/project/s1053/baroclinic_initialization/"
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILDENV_DIR=$JENKINS_DIR/../buildenv
PACE_DIR=$JENKINS_DIR/../

# setup module environment and default queue
test -f ${BUILDENV_DIR}/machineEnvironment.sh || exitError 1201 ${LINENO} "cannot find machineEnvironment.sh script"
. ${BUILDENV_DIR}/machineEnvironment.sh

set -x
. ${BUILDENV_DIR}/env.${host}.sh

# load scheduler tools
. ${BUILDENV_DIR}/schedulerTools.sh
scheduler_script="${BUILDENV_DIR}/submit.${host}.${scheduler}"

cp ${scheduler_script} job_initialization.sh
scheduler_script=job_initialization.sh

if grep -q "ranks" <<< "${experiment}"; then
    export NUM_RANKS=`echo ${experiment} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`
    echo "Setting NUM_RANKS=${NUM_RANKS}"
    if [ -f ${scheduler_script} ] ; then
        sed -i 's|<NTASKS>|<NTASKS>\n#SBATCH \-\-hint=multithread\n#SBATCH --ntasks-per-core=2|g' ${scheduler_script}
        if [ "$NUM_RANKS" -gt "6" ] && [ ! -v LONG_EXECUTION ]; then
            sed -i 's|cscsci|debug|g' ${scheduler_script}
        elif [ "$NUM_RANKS" -gt "6" ]; then
            sed -i 's|cscsci|normal|g' ${scheduler_script}
        fi
        sed -i 's|<NTASKS>|"'${NUM_RANKS}'"|g' ${scheduler_script}
        sed -i 's|<NTASKSPERNODE>|"24"|g' ${scheduler_script}
    fi
fi

export VIRTUALENV=${JENKINS_DIR}/../venv_driver
${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
source ${VIRTUALENV}/bin/activate

CMD="srun python3 ${PACE_DIR}/driver/examples/baroclinic_init.py ${JENKINS_DIR}/driver_configs/${experiment}.yaml"
run_command "${CMD}" Job${action} ${scheduler_script} $minutes
if [ $? -ne 0 ] ; then
  exitError 1510 ${LINENO} "problem while executing script ${script}"
fi

module load sarus
sarus pull elynnwu/pace:latest
echo "####### generating figures..."
srun -C gpu --partition=debug --account=s1053 --time=00:05:00 sarus run --mount=type=bind,source=${PACE_DIR},destination=/work elynnwu/pace:latest python /work/driver/examples/plot_baroclinic_init.py /work/output.zarr ${experiment} pt -1
mkdir -p ${ARTIFACT_ROOT}/${experiment}
echo "####### moving figures..."
cp *.png ${ARTIFACT_ROOT}/${experiment}/.

# no errors encountered
echo "####### finished: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"
exit 0
