#!/bin/bash
set -e
set -x
ARGS="--backend=dawn:gtmc -v -s -rsx"
make tests TEST_ARGS="$ARGS"
make tests_mpi TEST_ARGS="$ARGS"
