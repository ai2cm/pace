#!/bin/bash
set -e -x
for dataset in c12_6ranks_standard c12_54ranks_standard c128_6ranks_baroclinic ; do
    TEST_DATA_HOST="${TEST_DATA_DIR}/${dataset}/" make get_test_data
    mv ${TEST_DATA_HOST}/${dataset}.yml ${TEST_DATA_HOST}/input.yml
done
