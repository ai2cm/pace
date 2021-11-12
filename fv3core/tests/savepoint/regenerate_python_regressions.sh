#!/bin/bash

cd "`dirname $0`/../"

# generate list of experiments (and abort if none found)
set +e
EXPERIMENTS=`make list_test_data_options 2> /dev/null`
echo ${EXPERIMENTS}
set -e
if [ -z "${EXPERIMENTS}" ] ; then
    echo "Error: No matching experiments for the identified data version"
    exit 1
fi
BACKENDS=( numpy )
# loop over experiments
for experiment in ${EXPERIMENTS} ; do
  if [[ ! "$experiment" =~ ^gs\:\/\/ ]]; then
      continue
  fi
  exp_name=`basename ${experiment}`
  if [[ ! "$exp_name" =~ ^c ]]; then
      continue
  fi
  export NUM_RANKS=`echo ${exp_name} | grep -o -E '[0-9]+ranks' | grep -o -E '[0-9]+'`
  echo "====================================================="
  echo "Generating data for ${exp_name} ..."
  python_data_dir=test_data/${exp_name}/python_regressions
  if [[ ! -d ${python_data_dir} ]] ; then
      set +e
      echo "Making python regression data for the first time for ${exp_name}"
  fi
  for backend in ${BACKENDS} ; do
      echo "RUNNNING for backend ${backend}"
      EXPERIMENT=${exp_name} make savepoint_tests_mpi TEST_ARGS="--python_regression --force-regen --backend=${backend}" || true
  done
  sudo chown -R $USER:$USER ${python_data_dir}
  set -e
  # TODO: uncomment this if you want to update the official regressions
  #EXPERIMENT=${exp_name} make push_python_regressions
  echo "====================================================="
  echo ""
done
exit 0
