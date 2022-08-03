#!/bin/bash

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PACE_IMAGE="driver_image" make -C ${JENKINS_DIR}/.. build
docker run --rm driver_image make -C /pace/driver test test_mpi
