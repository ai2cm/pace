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
#SBATCH --ntasks=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=compile.out
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=normal


########################################################

set -x
export OMP_NUM_THREADS=12
export GT_CACHE_DIR_NAME=/tmp
srun python ${PACE_DIR}/driver/examples/compile_driver.py ${JENKINS_DIR}/driver_configs/compile_baroclinic_c576_54ranks.yaml ${PACE_DIR}
EOF

launch_job compile.daint.slurm 15000

cat << EOF > run.daint.slurm
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=run_pace
#SBATCH --ntasks=54
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.out
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=normal


########################################################

set -x
export OMP_NUM_THREADS=12
export GT_CACHE_DIR_NAME=/tmp
srun python ${PACE_DIR}/driver/examples/compile_driver.py ${JENKINS_DIR}/driver_configs/run_baroclinic_c576_54ranks.yaml ${PACE_DIR}
EOF
launch_job run.daint.slurm 15000

module load sarus
sarus pull elynnwu/pace:latest
echo "####### generating figures..."
srun -C gpu --partition=debug --account=s1053 --time=00:30:00 sarus run --mount=type=bind,source=${PACE_DIR},destination=/work elynnwu/pace:latest python /work/driver/examples/plot_pcolormesh_cube.py moist_baroclinic_c576 ua 40 --zarr_output=/work/output.zarr --diff_init --start=1 --stop=4 --force_symmetric_colorbar
echo "####### figures completed."