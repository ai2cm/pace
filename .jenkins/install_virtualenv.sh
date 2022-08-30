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
python3 -m pip install -r requirements_dev.txt -c constraints.txt

deactivate
