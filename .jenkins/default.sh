#!/bin/bash
set -e
set -x
make tests
make tests_mpi
