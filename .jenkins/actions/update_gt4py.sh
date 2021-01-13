#!/bin/bash
set -e
set -x
# Run when we change the gt4py source code
export CUDA=y
make pull_environment
make -C docker build_gt4py
make gt4py_tests_gpu
make tests
make tests_mpi
make -C docker push_gt4py
