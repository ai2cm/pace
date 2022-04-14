#!/usr/bin/env bash
# This example shows how to use CachingCommWriter to write MPI communication data to disk, and then re-run the model using the data from disk.

set -e -x

MPIRUN_CMD=${MPIRUN_CMD:-mpirun -n 6}

$MPIRUN_CMD python3 -m pace.driver.run configs/baroclinic_c12_comm_write.yaml --log-rank 0
python3 -m pace.driver.run configs/baroclinic_c12_comm_read.yaml
