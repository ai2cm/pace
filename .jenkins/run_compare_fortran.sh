#!/bin/bash

## Summary:
# Jenkins plan (only working on Piz daint) to run compare a run of pace with the fortran reference.
# This will install a virtualenv - make sure submodules are up to date

## Syntax:
# .jenkins/action/run_compare_fortran.sh <GridType> <InitType>

## Arguments:
# GridType: OneOf: serialbox, pace
# InitType: OneOf: serialbox, pace

GRID=$1
INIT=$2
JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$JENKINS_DIR/../
export VIRTUALENV=${PACE_DIR}/venv
${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
source ${VIRTUALENV}/bin/activate

BUILDENV_DIR=$PACE_DIR/buildenv
. ${BUILDENV_DIR}/schedulerTools.sh

cd $PACE_DIR

if [[ "${GRID}" == "serialbox" ]] && [[ "${INIT}" == "serialbox" ]]; then
	NAMELIST=baroclinic_c48_6ranks_dycore_only_serialbox
    sed -i "s/grid_option/true/g" ${JENKINS_DIR}/driver_configs/${NAMELIST}.yaml
elif [[ "${GRID}" == "pace" ]] && [[ "${INIT}" == "serialbox" ]]; then
    NAMELIST=baroclinic_c48_6ranks_dycore_only_serialbox
    sed -i "s/grid_option/false/g" ${JENKINS_DIR}/driver_configs/${NAMELIST}.yaml
elif [[ "${GRID}" == "pace" ]] && [[ "${INIT}" == "pace" ]]; then
    NAMELIST=baroclinic_c48_6ranks_dycore_only
else
    echo "This grid and init option is not supported."
    exit 1
fi


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
#SBATCH --account=go31
#SBATCH --partition=normal
########################################################
set -x
export OMP_NUM_THREADS=12
srun python -m pace.driver.run ${JENKINS_DIR}/driver_configs/${NAMELIST}.yaml
EOF

launch_job run.daint.slurm 2400
tar -czvf ${PACE_DIR}/archive.tar.gz ${PACE_DIR}/output.zarr


mkdir reference_data
cp -r /project/s1053/fortran_output/wrapper_output/c48_6ranks_baroclinic reference_data/c48_6ranks_baroclinic

module load sarus
sarus pull elynnwu/pace:latest
srun -C gpu --partition=normal --account=s1053 --time=00:30:00 sarus run --mount=type=bind,source=${PACE_DIR},destination=/work elynnwu/pace:latest python /work/driver/examples/plot_pcolormesh_cube.py dry_baro_c48_FtnRef_G_${GRID:0:1}_I_${INIT:0:1} ua 40 --start=0 --stop=20 --zarr_output=/work/output.zarr --fortran_data_path=/work/reference_data/c48_6ranks_baroclinic --fortran_var=eastward_wind --fortran_from_wrapper --size=48 --force_symmetric_colorbar
