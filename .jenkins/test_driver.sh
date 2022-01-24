#!/bin/bash

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

FV3GFS_IMAGE="driver_image" make -C ${JENKINS_DIR}/../docker fv3gfs_image
docker run --rm driver_image make -C /driver test test_mpi
