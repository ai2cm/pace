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
pace_dir=`dirname $0`/../
echo "pace path is ${pace_dir}"
if [ "$WHEEL_DIR" != "" ]; then
    wheel_command="--find-links=$WHEEL_DIR"
else
    wheel_command=""
fi
virtualenv_path=$1

set -e -x

git submodule update --init ${pace_dir}/external/daint_venv
git submodule update --init ${pace_dir}/external/gt4py
${pace_dir}/external/daint_venv/install.sh ${virtualenv_path}
source ${virtualenv_path}/bin/activate
python3 -m pip install ${PACE_INSTALL_FLAGS} ${pace_dir}/external/gt4py/
python3 -m pip install ${pace_dir}/pace-util/
python3 -m pip install $wheel_command -c ${pace_dir}/constraints.txt -r fv3core/requirements/requirements_base.txt
python3 -m pip install ${PACE_INSTALL_FLAGS} ${pace_dir}/fv3core/
python3 -m pip install ${PACE_INSTALL_FLAGS} ${pace_dir}/fv3gfs-physics/
python3 -m pip install ${PACE_INSTALL_FLAGS} ${pace_dir}/stencils/
python3 -m pip install ${PACE_INSTALL_FLAGS} ${pace_dir}/dsl/
deactivate
