#!/usr/bin/env bash
# This example shows how to use CachingCommWriter to write MPI communication data to disk, and then re-run the model using the data from disk.

set -e -x

if [ $# -eq 1 ]; then
    MPIRUN_CMD=$1
else
    MPIRUN_CMD="mpirun -n 6"
fi


$MPIRUN_CMD python3 -m pace.driver.run configs/baroclinic_c12_comm_write.yaml
python3 -m pace.driver.run configs/baroclinic_c12_comm_read.yaml
