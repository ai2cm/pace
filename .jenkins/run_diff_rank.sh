#!/bin/bash
JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PACE_DIR=$JENKINS_DIR/../
export VIRTUALENV=${PACE_DIR}/venv
# export FV3CORE_INSTALL_FLAGS="-e"
# export PACE_INSTALL_FLAGS="-e"
# ${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
source ${VIRTUALENV}/bin/activate

BUILDENV_DIR=$PACE_DIR/buildenv
. ${BUILDENV_DIR}/schedulerTools.sh

cd $PACE_DIR

mkdir -p 54_rank_job
cd 54_rank_job
# ${JENKINS_DIR}/fetch_caches.sh gt:gpu c192_54ranks_baroclinic driver

cat << EOF > run.daint.slurm
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=192_54_driver
#SBATCH --ntasks=54
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=normal
########################################################
set -x
export OMP_NUM_THREADS=12
srun python -m pace.driver.run ${JENKINS_DIR}/driver_configs/baroclinic_c192_54ranks.yaml
EOF

launch_job run.daint.slurm 29000
# ${JENKINS_DIR}/generate_caches.sh gt:gpu c192_54ranks_baroclinic driver
# tar -czvf ${PACE_DIR}/54_rank_ouput.tar.gz output.zarr

cd $PACE_DIR
mkdir -p  6_rank_job
cd 6_rank_job
# ${JENKINS_DIR}/fetch_caches.sh gt:gpu c192_6ranks_baroclinic driver
cat << EOF > run.daint.slurm
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=192_6_driver
#SBATCH --ntasks=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=normal
########################################################
set -x
export OMP_NUM_THREADS=12
srun python -m pace.driver.run ${JENKINS_DIR}/driver_configs/baroclinic_c192_6ranks.yaml
EOF

launch_job run.daint.slurm 29000
# ${JENKINS_DIR}/generate_caches.sh gt:gpu c192_6ranks_baroclinic driver
# tar -czvf ${PACE_DIR}/6_rank_ouput.tar.gz output.zarr
cd $PACE_DIR


module load sarus
sarus pull elynnwu/pace:latest
echo "####### generating figures..."
cat << EOF > plot.slurm
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=plotting
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run.out
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=debug
########################################################
set -x
sarus run --mount=type=bind,source=/scratch/snx3000/tobwi/sbox/pace,destination=/work elynnwu/pace:latest python /work/driver/examples/plot_pcolormesh_cube.py zacoustics_c192_diff_numpy u 40 --zarr_output=/work/54_rank_job/output.zarr --vmin=-0.00001 --vmax=0.00001 --diff_python_path=/work/6_rank_job/output.zarr --size=192 --start=0 --stop=2
sarus run --mount=type=bind,source=/scratch/snx3000/tobwi/sbox/pace,destination=/work elynnwu/pace:latest python /work/driver/examples/plot_pcolormesh_cube.py zacoustics_c192_diff_numpy uc 40 --zarr_output=/work/54_rank_job/output.zarr --vmin=-0.00001 --vmax=0.00001 --diff_python_path=/work/6_rank_job/output.zarr --size=192 --start=0 --stop=2
EOF
launch_job plot.slurm 5000
echo "####### figures completed."
