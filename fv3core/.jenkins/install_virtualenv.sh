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
if [ "$WHEEL_DIR" != "" ]; then
    wheel_command="--find-links=$WHEEL_DIR"
else
    wheel_command=""
fi

git submodule update --init
virtualenv_path=$1
fv3core_dir=`dirname $0`/../
pace_dir=`dirname $0`/../../
${pace_dir}/external/daint_venv/install.sh ${virtualenv_path}
source ${virtualenv_path}/bin/activate
python3 -m pip install -r ${pace_dir}/requirements_dev.txt
deactivate
