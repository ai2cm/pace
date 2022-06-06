#!/usr/bin/env bash

backend=$1
experiment=$2

data_version=$(cd fv3core && EXPERIMENT=$experiment TARGET=driver make get_test_data | tail -1)
.jenkins/initialize_driver.py fv3core/test_data/$data_version/$experiment/driver $backend
