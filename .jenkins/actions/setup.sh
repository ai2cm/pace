#!/bin/bash
set -e -x

if [ ${python_env} == "virtualenv" ]; then
    if [ -d ${VIRTUALENV} ]; then
        echo "Using existing virtualenv ${VIRTUALENV}"
    else
        ${jenkins_dir}/install_virtualenv.sh ${VIRTUALENV}
    fi
fi
