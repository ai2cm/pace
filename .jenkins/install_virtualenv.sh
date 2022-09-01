#!/bin/bash

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}


# check a virtualenv path has been provided
test -n "$1" || exitError 1001 ${virtualenv_path} "must pass an argument"
JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$JENKINS_DIR/../
echo "pace path is ${PACE_DIR}"

if [ "$WHEEL_DIR" != "" ]; then
    wheel_command="--find-links=$WHEEL_DIR"
else
    wheel_command=""
fi
virtualenv_path=$1

set -e -x

workdir=$(pwd)
git submodule update --init ${PACE_DIR}/external/daint_venv
git submodule update --init ${PACE_DIR}/external/gt4py
${PACE_DIR}/external/daint_venv/install.sh ${virtualenv_path}
source ${virtualenv_path}/bin/activate

workdir=$(pwd)
cd ${PACE_DIR}
python3 -m pip install $wheel_command -r ${PACE_DIR}/requirements_dev.txt -c ${PACE_DIR}/constraints.txt
# have to be installed in non-develop mode because the directory where this gets built from
# gets deleted before the tests run on daint
python3 -m pip install ${PACE_DIR}/driver ${PACE_DIR}/dsl ${PACE_DIR}/fv3core ${PACE_DIR}/physics ${PACE_DIR}/stencils ${PACE_DIR}/util ${PACE_DIR}/external/gt4py -c ${PACE_DIR}/constraints.txt
cd ${workdir}

deactivate
