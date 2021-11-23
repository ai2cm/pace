#!/bin/bash

# setup environment for different systems
#
# NOTE: the location of the base bash script and module initialization
#       vary from system to system, so you will have to add the location
#       if your system is not supported below

exitError()
{
    \rm -f /tmp/tmp.${user}.$$ 1>/dev/null 2>/dev/null
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

showWarning()
{
    echo "WARNING $1: $3" 1>&2
    echo "WARNING       LOCATION=$0" 1>&2
    echo "WARNING       LINE=$2" 1>&2
}

modulepathadd() {
    if [ -d "$1" ] && [[ ":$MODULEPATH:" != *":$1:"* ]]; then
        MODULEPATH="${MODULEPATH:+"$MODULEPATH:"}$1"
    fi
}

# setup empty defaults
host=""          # name of host
scheduler=""     # none, slurm, pbs, ...
queue=""         # standard queue to submit jobs to
nthreads=""      # number of threads to use for parallel builds
mpilaunch=""     # command to launch an MPI executable (e.g. aprun)
installdir=""    # directory where libraries are installed
container_engine=""  # Engine for running containers, e.g. docker, sarus, singuilarity
python_env=""    # Preferred environment in which to run python code, e.g. virtualenv, container
# set default value for useslurm based on whether a submit script exists
envdir=`dirname $0`
# setup machine specifics
if [ "`hostname | grep daint`" != "" ] ; then
    . /etc/bash.bashrc
    . /opt/modules/default/init/bash
    . /etc/bash.bashrc.local
    export host="daint"
    scheduler="slurm"
    queue="normal"
    nthreads=8
    mpilaunch="srun"
    installdir=/project/s1053/install/
    container_engine="sarus"
    python_env="virtualenv"
    export CUDA_ARCH=sm_60
elif [ "`hostname | grep papaya`" != "" ] ; then
    alias module=echo
    export host="papaya"
    scheduler="none"
    queue="normal"
    nthreads=6
    mpilaunch="mpirun"
    installdir=/Users/OliverF/Desktop/install
    container_engine="docker"
    python_env="container"
elif [ "`hostname | grep ubuntu-1804`" != "" ] ; then
    . /etc/profile
    alias module=echo
    export host="gce"
    scheduler="none"
    queue="normal"
    nthreads=6
    mpilaunch="mpirun"
    installdir=/tmp
    container_engine="docker"
    python_env="container"
    if [ ! -z "`command -v nvidia-smi`" ] ; then
        nvidia-smi 2>&1 1>/dev/null
        if [ $? -eq 0 ] ; then
            export CUDA_ARCH=sm_60
        fi
    fi
elif [ "${CIRCLECI}" == "true" ] ; then
    alias module=echo
    export host="circleci"
    scheduler="none"
    queue="normal"
    nthreads=6
    mpilaunch="mpirun"
    installdir=/tmp
    container_engine="docker"
    python_env="container"
fi

# make sure everything is set
test -n "${host}" || exitError 2001 ${LINENO} "Variable <host> could not be set (unknown machine `hostname`?)"
test -n "${queue}" || exitError 2002 ${LINENO} "Variable <queue> could not be set (unknown machine `hostname`?)"
test -n "${scheduler}" || exitError 2002 ${LINENO} "Variable <scheduler> could not be set (unknown machine `hostname`?)"
test -n "${nthreads}" || exitError 2003 ${LINENO} "Variable <nthreads> could not be set (unknown machine `hostname`?)"
test -n "${mpilaunch}" || exitError 2004 ${LINENO} "Variable <mpilaunch> could not be set (unknown machine `hostname`?)"
test -n "${installdir}" || exitError 2005 ${LINENO} "Variable <installdir> could not be set (unknown machine `hostname`?)"
test -n "${container_engine}" || exitError 2005 ${LINENO} "Variable <container_engine> could not be set (unknown machine `hostname`?)"
test -n "${python_env}" || exitError 2005 ${LINENO} "Variable <python_env> could not be set (unknown machine `hostname`?)"
# export installation directory
export INSTALL_DIR="${installdir}"
