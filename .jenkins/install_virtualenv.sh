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

git submodule update --init ${PACE_DIR}/external/daint_venv
git submodule update --init ${PACE_DIR}/external/gt4py
${PACE_DIR}/external/daint_venv/install.sh ${virtualenv_path}
source ${virtualenv_path}/bin/activate
python3 -m pip install -r fv3core/requirements/requirements_dace.txt # To allow for non-release version of DaCe to be picked up first (since PIP doesn't allow git constrainted anymore...)
python3 -m pip install ${PACE_INSTALL_FLAGS} ${PACE_DIR}/external/gt4py/
python3 -m pip install ${PACE_DIR}/util/
python3 -m pip install $wheel_command -c ${PACE_DIR}/constraints.txt -r fv3core/requirements/requirements_base.txt
python3 -m pip install ${PACE_INSTALL_FLAGS} ${PACE_DIR}/fv3core/
python3 -m pip install ${PACE_INSTALL_FLAGS} ${PACE_DIR}/physics/
python3 -m pip install ${PACE_INSTALL_FLAGS} ${PACE_DIR}/stencils/
python3 -m pip install ${PACE_INSTALL_FLAGS} ${PACE_DIR}/dsl/
python3 -m pip install ${PACE_INSTALL_FLAGS} ${PACE_DIR}/driver/

deactivate
