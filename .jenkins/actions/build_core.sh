#!/bin/bash
set -e -x
export DOCKER_BUILDKIT=1
make pull_environment
make build  
make push_core
make tar_core
