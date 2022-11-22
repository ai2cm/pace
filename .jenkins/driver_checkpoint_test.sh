#!/bin/bash

## Summary:
# Jenkins plan (only working on Piz daint) to run dace orchestration and gather performance numbers.

## Syntax:
# .jenkins/action/driver_checkpoint_test.sh

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$JENKINS_DIR/../
export VIRTUALENV=${PACE_DIR}/venv
${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
source ${VIRTUALENV}/bin/activate

BUILDENV_DIR=$PACE_DIR/buildenv
. ${BUILDENV_DIR}/schedulerTools.sh

cat << EOF > run.daint.slurm
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=driver_checkpoint_test
#SBATCH --ntasks=6
#SBATCH --hint=multithread
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=1
#SBATCH --output=driver.out
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=normal
########################################################
set -x
export OMP_NUM_THREADS=1
export TEST_ARGS="-v -s -rsx --backend=numpy "
export EXPERIMENT=c12_6ranks_baroclinic_dycore_microphysics
export MPIRUN_CALL="srun"
CONTAINER_CMD="" MPIRUN_ARGS="" DEV=n make test_driver_checkpoint
EOF
launch_job run.daint.slurm 3600
