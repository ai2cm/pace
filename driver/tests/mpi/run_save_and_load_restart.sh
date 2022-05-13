#!/bin/bash
set -e
MPIRUN_CALL=${MPIRUN_CALL:-mpirun -n 6}
cp examples/configs/baroclinic_c12_write_restart.yaml baroclinic_c12_write_restart.yaml
cp examples/configs/baroclinic_c12_write_restart.yaml baroclinic_c12_run_two_steps.yaml
sed -i 's/seconds: 225/seconds: 450/' baroclinic_c12_run_two_steps.yaml
sed -i 's/save_restart: true/save_restart: false/' baroclinic_c12_run_two_steps.yaml
sed -i 's/path: "output.zarr"/path: "run_two_steps_output.zarr"/' baroclinic_c12_run_two_steps.yaml
$MPIRUN_CALL python -m pace.driver.run baroclinic_c12_write_restart.yaml
$MPIRUN_CALL python -m pace.driver.run RESTART/restart.yaml
$MPIRUN_CALL python -m pace.driver.run baroclinic_c12_run_two_steps.yaml
