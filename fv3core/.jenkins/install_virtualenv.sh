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

make update_submodules_venv
virtualenv_path=$1
fv3core_dir=`dirname $0`/../
${fv3core_dir}/external/daint_venv/install.sh ${virtualenv_path}
source ${virtualenv_path}/bin/activate
python3 -m pip install ${FV3CORE_INSTALL_FLAGS} ${fv3core_dir}/external/gt4py/ -c ${fv3core_dir}/constraints.txt
python3 -m pip install ${FV3CORE_INSTALL_FLAGS} ${fv3core_dir}/external/pace-util/ -c ${fv3core_dir}/../constraints.txt
python3 -m pip install ${FV3CORE_INSTALL_FLAGS} ${fv3core_dir}/external/stencils/ -c ${fv3core_dir}/../constraints.txt
python3 -m pip install ${FV3CORE_INSTALL_FLAGS} ${fv3core_dir}/external/dsl/ -c ${fv3core_dir}/../constraints.txt
python3 -m pip install $wheel_command -c ${fv3core_dir}/../constraints.txt -r ${fv3core_dir}/requirements/requirements_base.txt
python3 -m pip install ${FV3CORE_INSTALL_FLAGS} ${fv3core_dir} -c ${fv3core_dir}/../constraints.txt
deactivate
