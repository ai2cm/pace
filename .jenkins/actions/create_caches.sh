#!/usr/bin/env bash

backend=$1
experiment=$2
[[ "$3" != "bypass_wrapper" ]] && parallel_prefix="srun"
target=driver

data_version=$(cd fv3core && EXPERIMENT=$experiment TARGET=$target make get_test_data | tail -1)

# Jenkins sets TEST_DATA_ROOT to a custom path and `make get_test_data` uses that above
data_path=${TEST_DATA_ROOT:-fv3core/test_data}/$data_version/$experiment/$target

$parallel_prefix .jenkins/initialize_caches.py $data_path $backend
