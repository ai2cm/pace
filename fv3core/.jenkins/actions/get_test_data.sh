#!/bin/bash
set -e -x
for dataset in c12_6ranks_standard c12_54ranks_standard c128_6ranks_baroclinic ; do
    EXPERIMENT=${dataset} make get_test_data
done
