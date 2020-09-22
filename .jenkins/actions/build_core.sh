#!/bin/bash
set -e -x
export DOCKER_BUILDKIT=1
make pull_environment
export FV3_TAG="${JOB_NAME}-${BUILD_NUMBER}"
make build
make push_core
make tar_core
