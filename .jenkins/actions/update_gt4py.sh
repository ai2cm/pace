#!/bin/bash
set -e
set -x
# Run when we change the gt4py source code

make pull_environment
make -C docker build_gt4py
make tests
make tests_mpi
make -C docker push_gt4py
