#!/bin/bash
JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$JENKINS_DIR/../
export VIRTUALENV=${PACE_DIR}/venv
${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
source ${VIRTUALENV}/bin/activate

BUILDENV_DIR=$PACE_DIR/buildenv
. ${BUILDENV_DIR}/schedulerTools.sh

cd $PACE_DIR
cat << EOF > compile.daint.slurm
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=compile_driver
#SBATCH --ntasks=54
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=compile.out
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=normal


########################################################

set -x
export OMP_NUM_THREADS=12
export FV3_DACEMODE=Build
srun python -m pace.driver.run ${JENKINS_DIR}/driver_configs/baroclinic_c576_54ranks.yaml
EOF

launch_job compile.daint.slurm 45000

cat << EOF > run.daint.slurm
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=run_pace
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.out
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=normal


########################################################

set -x
export OMP_NUM_THREADS=12
export FV3_DACEMODE=Run
srun python -m pace.driver.run ${JENKINS_DIR}/driver_configs/baroclinic_c768_96ranks.yaml
EOF
launch_job run.daint.slurm 15000

tar -czvf ${PACE_DIR}/output.tar.gz output.zarr

