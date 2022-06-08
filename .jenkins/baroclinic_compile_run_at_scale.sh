#!/bin/bash

# Jenkins action to compile baroclinic test case then run at larger scale on Piz Daint

# Syntax:
# .jenkins/action/baroclinic_compile_run_at_scale.she

## Arguments:

# utility function for error handling
exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

# stop on all errors and echo commands
set -x +e

ARTIFACT_ROOT="/project/s1053/baroclinic_compile_run_at_scale/"
echo "####### executing: $0 $* (PID=$$ HOST=$HOSTNAME TIME=`date '+%D %H:%M:%S'`)"

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILDENV_DIR=$JENKINS_DIR/../buildenv
PACE_DIR=$JENKINS_DIR/../

# setup module environment and default queue
test -f ${BUILDENV_DIR}/machineEnvironment.sh || exitError 1201 ${LINENO} "cannot find machineEnvironment.sh script"
. ${BUILDENV_DIR}/machineEnvironment.sh

set -x
. ${BUILDENV_DIR}/env.${host}.sh

# load scheduler tools
. ${BUILDENV_DIR}/schedulerTools.sh

# export VIRTUALENV=${JENKINS_DIR}/../venv_driver
# ${JENKINS_DIR}/install_virtualenv.sh ${VIRTUALENV}
# source ${VIRTUALENV}/bin/activate

# First, generate 9 caches from a 3x3 layout namelist
cat > compile.slurm <<EOF
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=pace_compile_54ranks
#SBATCH --ntasks=54
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=compile_54.out
#SBATCH --time=02:30:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=normal


########################################################

set -x
export OMP_NUM_THREADS=12
srun python3 -m pace.driver.run ${JENKINS_DIR}/driver_configs/baroclinic_c576_54ranks.yaml
EOF

launch_job compile.slurm 9500

# Now, remove caches 9-54
rm -rf .gt_cache_000009
rm -rf .gt_cache_00001*
rm -rf .gt_cache_00002*
rm -rf .gt_cache_00003*
rm -rf .gt_cache_00004*
rm -rf .gt_cache_00005*

# Then use the 9 caches to run for a 4x4 layout
mkdir -p .layout
cat > .layout/decompostiion.yaml << EOF
'00': 0
'01': 3
'02': 6
'10': 1
'11': 4
'12': 7
'20': 2
'21': 5
'22': 8
layout:
- 3
- 3
EOF

cat > run_at_scale.slurm <<EOF
#!/bin/bash
#SBATCH --constraint=gpu
#SBATCH --job-name=pace_run_at_scale
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=run_96.out
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=s1053
#SBATCH --partition=normal


########################################################

set -x
export OMP_NUM_THREADS=12
export RUN_ONLY=TRUE
export PYTHONOPTIMIZE=TRUE
srun python3 -m pace.driver.run ${JENKINS_DIR}/driver_configs/baroclinic_c768_96ranks.yaml
EOF

launch_job run_at_scale.slurm 3900

# Now plot the results
module load sarus
sarus pull elynnwu/pace:latest
srun -C gpu --partition=debug --account=s1053 --time=00:30:00 sarus run --mount=type=bind,source=$PACE_DIR,destination=/work --mount=type=bind,source=${JENKINS_DIR}/run_at_larger_scale,destination=/pace elynnwu/pace:latest python /work/driver/examples/plot_pace_zarr_output.py Diff_init_pace_c768_96ranks ua 40 --size=768 --start=0 --stop=6 --diff_init --force_symmetric_colorbar
echo "####### moving figures..."
cp $PACE_DIR/Diff_init_pace_c768_96ranks*.png ${ARTIFACT_ROOT}/.