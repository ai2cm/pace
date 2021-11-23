#!/bin/bash

# CONTAINER tools (e.g. docker, sarus, singularity)

##################################################
# functions
##################################################
function get_container {
    local REMOTE_LOCATION=$1
    if [ "${container_engine}" == "docker" ] ; then
	docker pull ${REMOTE_LOCATION}
    fi
    if [ "${container_machine}" == "sarus" ] ; then
	TARFILE=`basename -- ${REMOTE_LOCATION}`
	if [ ! -f ${TARFILE} ] ; then
	    gsutil copy ${REMOTE_LOCATION} .
	fi
	CONTAINER=${TARFILE%.tar}
	sarus load ./$(TARFILE) ${CONTAINER}
    fi
}

function run_container {
    local IMAGE=$1
    local CMD=$2
    local FLAGS=$3
    local PARALLEL=$4
    if [ "${container_engine}" == "docker" ] ; then
	local FLAGS="--rm ${FLAGS}"
    fi
    if  [ "${container_engine}" == "sarus" ] ; then
	local IMAGE=load/library/${IMAGE}
	if ${PARALLEL} ; then
	    local PARALLEL_RUN="srun"
	    local FLAGS="--mpi ${FLAGS}"
	fi
    fi

    ${PARALLEL_RUN} ${container_engine} run ${FLAGS} "${IMAGE}" "${CMD}"
}
