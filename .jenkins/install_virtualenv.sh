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

virtualenv_path=$1
root=`dirname $0`/../
(cd ${root}/external/daint_venv && ./install.sh ${virtualenv_path})
source ${virtualenv_path}/bin/activate
python3 -m pip install ${root}/external/fv3gfs-util/
python3 -m pip install -c ${root}/constraints.txt -r ${root}/requirements.txt
python3 -m pip install ${root}
deactivate
