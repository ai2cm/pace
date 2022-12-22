#!/bin/bash

## Summary:
# Jenkins plan (only working on Piz daint) to run dace orchestration and gather performance numbers.

## Syntax:
# .jenkins/action/driver_performance.sh

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$JENKINS_DIR/../
export VIRTUALENV=${PACE_DIR}/venv
${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
source ${VIRTUALENV}/bin/activate

BUILDENV_DIR=$PACE_DIR/buildenv
. ${BUILDENV_DIR}/schedulerTools.sh

mkdir -p ${PACE_DIR}/test_perf
cd $PACE_DIR/test_perf
cat << EOF > run.daint.slurm
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=c192_pace_driver
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=driver.out
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --account=go31
#SBATCH --partition=normal
########################################################
set -x
export OMP_NUM_THREADS=12
export FV3_DACEMODE=BuildAndRun
srun python -m pace.driver.run ${JENKINS_DIR}/driver_configs/baroclinic_c192_6ranks.yaml
EOF
launch_job run.daint.slurm 3600

python ${JENKINS_DIR}/print_performance_number.py
cp *.json driver.out /project/s1053/performance/fv3core_performance/dace_gpu
