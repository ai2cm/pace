#!/usr/bin/env bash

backend=$1
experiment=$2
[[ "$3" != "bypass_wrapper" ]] && parallel_prefix="srun"

data_version=$(cd fv3core && EXPERIMENT=$experiment TARGET=driver make get_test_data | tail -1)
$parallel_prefix .jenkins/initialize_caches.py fv3core/test_data/$data_version/$experiment/driver $backend
