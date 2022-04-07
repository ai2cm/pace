#!/bin/bash
BACKEND=$1
EXPNAME=$2


python_data_dir=${TEST_DATA_ROOT}/python_regressions
if [[ ! -d ${python_data_dir} ]] ; then
    set +e
    echo "Making python regression data for the first time for {exp_name}"
fi
export EXPERIMENT=${EXPNAME}

# sync the test data
make get_test_data

# Run the tests to generate the python regressions
make savepoint_tests_mpi TEST_ARGS="--python_regression --force-regen --backend=${BACKEND}" || true
sudo chown -R $USER:$USER ${python_data_dir}
set -e
EXPERIMENT=${exp_name} make push_python_regressions
