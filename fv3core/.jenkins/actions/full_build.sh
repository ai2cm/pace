#!/bin/bash
set -e
set -x
# Run when dependency.Dockerfile changes the environment image, serialbox or mpich
export PULL=False
make savepoint_tests
make savepoint_tests_mpi
make push_environment
make build_cuda_environment
CUDA=y make push_environment
