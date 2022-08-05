#!/bin/bash
JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$JENKINS_DIR/../
export VIRTUALENV=${PACE_DIR}/venv
${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
source ${VIRTUALENV}/bin/activate

BUILDENV_DIR=$PACE_DIR/buildenv
. ${BUILDENV_DIR}/schedulerTools.sh

cd $PACE_DIR


${JENKINS_DIR}/fetch_caches.sh "gt:gpu" "c48_6ranks_baroclinic" dycore

cat << EOF > run.daint.slurm
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=c48_driver
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=c48_driver.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=debug
########################################################
set -x
export OMP_NUM_THREADS=12
srun python -m pace.driver.run ${JENKINS_DIR}/driver_configs/baroclinic_c48_6ranks_dycore_only.yaml
EOF

launch_job run.daint.slurm 2400
module load sarus
sarus pull elynnwu/pace:latest
srun -C gpu --partition=debug --account=s1053 --time=00:30:00 sarus run --mount=type=bind,source=${PACE_DIR},destination=/work elynnwu/pace:latest python /work/driver/examples/plot_pcolormesh_cube.py dry_baroclinic_c48_comparison ua 40 --start=0 --stop=20 --zarr_output=/work/output.zarr --fortran_data_path=/project/s1053/fortran_output/wrapper_output/c48_6ranks_baroclinic --fortran_var=eastward_wind --fortran_from_wrapper
