#!/bin/bash

# Jenkins action to run a large simulation on Piz Daint

# utility functions
function exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}
function run_job {
    batch_script=$1
    set +e
    res=$(sbatch -W -C gpu $batch_script 2>&1)
    status1=$?
    grep -q SUCCESS $batch_script
    status2=$?
    set -e
    wait
    echo "DONE WAITING ${status1} ${status2}"
    if [ $status1 -ne 0 -o $status2 -ne 0 ] ; then
	echo "ERROR: run not sucessful"
	head -400 *.out
	exit 1
    else
	echo $batch_script
	echo "run sucessful"
    fi
}
set -e

# configuration
SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$SCRIPTPATH")")"
BUILDENV_DIR="$ROOT_DIR/../buildenv"
githash=`git rev-parse HEAD`

# check sanity of environment
test -n "$1" || exitError 1001 ${LINENO} "must pass a number of ranks to compile with"
test -n "$2" || exitError 1001 ${LINENO} "must pass a number of tile edge points to compile with"
test -n "$3" || exitError 1002 ${LINENO} "must pass a number of ranks to run at scale with"
test -n "$4" || exitError 1001 ${LINENO} "must pass a number of tile edge points to run at scale with"
test -n "$5" || exitError 1003 ${LINENO} "must pass a backend"
test -n "$6" || exitError 1004 ${LINENO} "must pass a namelists root directory"

ranks_compile=$1
size_compile=$2
ranks_scale=$3
size_scale=$4
backend=$5
namelists_root_dir=$6
env_vars="export PYTHONOPTIMIZE=TRUE\nexport CRAY_CUDA_MPS=0\nexport FV3_STENCIL_REBUILD_FLAG=False"
compile_script=daint.${ranks_compile}ranks.slurm
scale_script=daint.${ranks_scale}ranks.slurm
namelist_folder_compile=${namelist_root_dir}/c${size_compile}_${ranks_compile}_baroclinic

cp $BUILDENV_DIR/submit.daint.slurm $compile_script

# Adapt batch script to run the compilation:
sed -i "s/<NAME>/scale-compilation/g" $compile_script
sed -i "s/<NTASKS>/${ranks_compile}/g" $compile_script
sed -i "s/<NTASKSPERNODE>/1/g" $compile_script
sed -i "s/<CPUSPERTASK>/$NTHREADS/g" $compile_script
sed -i "s/<OUTFILE>/run.daint${ranks_compile}.out\n#SBATCH --hint=nomultithread/g" $compile_script
sed -i "s/00:45:00/03:15:00/g" $compile_script
sed -i "s/cscsci/normal/g" $compile_script
sed -i "s#<CMD>#$env_vars\nsrun python examples/standalone/runfile/dynamics.py $namelist_folder_compile 2 $backend $githash #g" $compile_script

# Copy compile script and adjust for running at scale settings
cp $compile_script $scale_script
sed -i "s/scale-compilation/scale/g" $scale_script
sed -i "s/${ranks_compile}/${ranks_scale}/g" $compile_script
sed -i "s/c${size_compile}/c${size_scale}/g" $compile_script
sed -i "s/03:15:00/02:00:00/g" $compile_script

# set up the virtual environment
$ROOT_DIR/.jenkins/install_virtualenv.sh $ROOT_DIR/venv
source ./venv/bin/activate
pip list
# create the cache
run_job $compile_script
# run at scale
run_job $scale_script

