#!/bin/bash

#############################################
# Example syntax:
# ./run_on_daint.sh 60 6 gtx86 <path_to_serialized_data>

## Arguments:
# $1: number of timesteps to run
# $2: number of ranks to execute with (ensure that this is compatible with fv3core)
# $3: backend to use in gt4py
# $4: path to the data directory that should be run
# $5: (optional) arguments to pass to python invocation
# $6: (optional) arguments to pass to dynamics.py invocation
# $7: (optional) true|false wraps an extra 2 timestep run in nsys

# stop on all errors
set -e

# configuration
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
FV3CORE_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
BUILDENV_DIR="$FV3CORE_DIR/../buildenv"
PACE_DIR=$SCRIPT_DIR/../../../../
NTHREADS=12


# utility functions
function exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

function cleanupFailedJob {
    res=$1
    jobout=$2
    jobid=`echo "${res}" | sed  's/^Submitted batch job //g'`
    test -n "${jobid}" || exitError 7207 ${LINENO} "problem determining job ID of SLURM job"
    echo "jobid:" ${jobid}
    status=`sacct --jobs ${jobid} -X -p -n -b -D `
    while [[ $status == *"COMPLETING"* ]]; do
        if [ $timeout -lt 120 ]; then
	    status=`sacct --jobs ${jobid} -p -n -b -D `
	    sleep 30
	    timeout=$timeout + 30
        else
            exitError 1004 ${LINENO} "problem waiting for job ${jobid} to complete"
        fi
    done
    if [[ ! $status == *"COMPLETED"* ]]; then
        echo ${status}
        echo `cat ${jobout}`
        rm -rf .gt_cache*
        pip list
        deactivate
        rm -rf venv
        exitError 1003 ${LINENO} "problem in slurm job"
    fi
}

# check sanity of environment
test -n "$1" || exitError 1001 ${LINENO} "must pass a number of timesteps"
timesteps="$1"
test -n "$2" || exitError 1002 ${LINENO} "must pass a number of ranks"
ranks="$2"
test -n "$3" || exitError 1003 ${LINENO} "must pass a backend"
backend="$3"
test -n "$4" || exitError 1004 ${LINENO} "must pass a data path"
data_path="$4"
py_args="$5"
run_args="$6"
DO_NSYS_RUN="$7"


# get dependencies
cd $FV3CORE_DIR
git submodule update --init


# set up the virtual environment
echo "creating the venv"
if [ -d ./venv ] ; then rm -rf venv ; fi
if [ -d ./gt4py ] ; then rm -rf gt4py ; fi
cd $FV3CORE_DIR
$PACE_DIR/.jenkins/install_virtualenv.sh $FV3CORE_DIR/venv
source ./venv/bin/activate
pip list

# set the environment
cp $BUILDENV_DIR/submit.daint.slurm compile.daint.slurm
cp $BUILDENV_DIR/submit.daint.slurm run.daint.slurm
if [ "${DO_NSYS_RUN}" == "true" ] ; then
    cp $BUILDENV_DIR/submit.daint.slurm run.nsys.daint.slurm
fi

if git rev-parse --git-dir > /dev/null 2>&1 ; then
  githash=`git rev-parse HEAD`
else
  githash="notarepo"
fi

split_path=(${data_path//\// })
experiment=${split_path[-1]}

echo "Configuration overview:"
echo "    Root dir:          $FV3CORE_DIR"
echo "    Timesteps:         $timesteps"
echo "    Ranks:             $ranks"
echo "    Backend:           $backend"
echo "    Input data dir:    $data_path"
echo "    Experiment:        $experiment"
echo "    Threads per rank:  $NTHREADS"
echo "    GIT hash:          $githash"
echo "    Python arguments:  $py_args"
echo "    Run arguments:     $run_args"
echo "    Extra run in nsys: $DO_NSYS_RUN"


$FV3CORE_DIR/../.jenkins/fetch_caches.sh $backend $EXPNAME dycore
tile_size=$(( ranks / 6 ))
COMPILING_RANKS=$(( 3 < tile_size ? 3 : tile_size ))
CPU_PER_TASK=$(( 12 / COMPILING_RANKS ))
env_vars="export PYTHONOPTIMIZE=TRUE\nexport CRAY_CUDA_MPS=1"
clear_cache="rm -rf $FV3CORE_DIR/.gt_cache*"
set_up_cache_dir="export GT_CACHE_DIR_NAME=/tmp"
execution="srun python examples/standalone/runfile/compile.py $data_path $backend $FV3CORE_DIR"
run_command="$env_vars\n$clear_cache\n$set_up_cache_dir\n$execution"
# Adapt batch script to compile the code:
sed -i "s/<NAME>/compilestandalone/g" compile.daint.slurm
sed -i "s/<NTASKS>/$COMPILING_RANKS/g" compile.daint.slurm
sed -i "s/<NTASKSPERNODE>/$COMPILING_RANKS/g" compile.daint.slurm
sed -i "s/<CPUSPERTASK>/$CPU_PER_TASK/g" compile.daint.slurm
sed -i "s/<OUTFILE>/compile.daint.out\n#SBATCH --hint=nomultithread/g" compile.daint.slurm
sed -i "s/<TIMEOUT>/03:00:00/g" compile.daint.slurm
sed -i "s#<CMD>#$run_command#g" compile.daint.slurm


env_vars="export PYTHONOPTIMIZE=TRUE\nexport CRAY_CUDA_MPS=0"
# Adapt batch script to run the code:
sed -i "s/<NAME>/standalone/g" run.daint.slurm
sed -i "s/<NTASKS>/$ranks/g" run.daint.slurm
sed -i "s/<NTASKSPERNODE>/1/g" run.daint.slurm
sed -i "s/<CPUSPERTASK>/$NTHREADS/g" run.daint.slurm
sed -i "s/<OUTFILE>/run.daint.out\n#SBATCH --hint=nomultithread/g" run.daint.slurm
sed -i "s/<TIMEOUT>/00:45:00/g" run.daint.slurm
sed -i "s/cscsci/normal/g" run.daint.slurm
sed -i "s#<CMD>#$env_vars\nsrun python $py_args examples/standalone/runfile/dynamics.py $data_path $timesteps $backend $githash $run_args#g" run.daint.slurm

echo "submitting script to do compilation"
set +e
res=$(sbatch -W -C gpu compile.daint.slurm 2>&1)
status1=$?
grep -q SUCCESS compile.daint.out
status2=$?
set -e
wait
echo "DONE WAITING ${status1} ${status2}"
if [ $status1 -ne 0 -o $status2 -ne 0 ] ; then
    cleanupFailedJob "${res}" compile.daint.out
    echo "ERROR: compilation step failed"
    exit 1
else
    echo "compilation step finished"
fi

echo "Submitting script to do performance run"
set +e
res=$(sbatch -W -C gpu run.daint.slurm 2>&1)
status1=$?
grep -q SUCCESS run.daint.out
status2=$?
set -e
wait
echo "DONE WAITING ${status1} ${status2}"
if [ $status1 -ne 0 -o $status2 -ne 0 ] ; then
    if [ ! -z "${BUILD_TAG}" ] ; then
	cleanupFailedJob "${res}" run.daint.out
    fi
    echo "ERROR: performance run not sucessful"
    exit 1
else
    echo "performance run sucessful"
fi

if [ "${DO_NSYS_RUN}" == "true" ] ; then
    module load nvidia-nsight-systems/2021.1.1.66-6c5c5cb
    echo "Install performance_visualization package"
    git clone git@github.com:ai2cm/performance_visualization.git
    pip install -e performance_visualization.git
    echo "submitting script to do performance run wrapped by nsys"
    # Adapt batch script to run the code:
    sed -i "s/<NAME>/standalone/g" run.nsys.daint.slurm
    sed -i "s/<NTASKS>/$ranks/g" run.nsys.daint.slurm
    sed -i "s/<NTASKSPERNODE>/1/g" run.nsys.daint.slurm
    sed -i "s/<CPUSPERTASK>/$NTHREADS/g" run.nsys.daint.slurm
    sed -i "s/<OUTFILE>/run.nsys.daint.out\n#SBATCH --hint=nomultithread/g" run.nsys.daint.slurm
    sed -i "s/<TIMEOUT>/00:40:00/g" run.nsys.daint.slurm
    sed -i "s/cscsci/normal/g" run.nsys.daint.slurm
    sed -i "s#<CMD>#srun nsys profile --force-overwrite=true -o %h.%q{SLURM_NODEID}.%q{SLURM_PROCID}.qdstrm --trace=cuda,mpi,nvtx --mpi-impl=mpich python ./performance_visualization/analysis/pywrapper.py --config ./performance_visualization/config_examples/f3core.json --nvtx examples/standalone/runfile/dynamics.py $data_path 3 $backend $githash --disable_json_dump#g" run.nsys.daint.slurm
    # execute on a gpu node
    set +e
    res=$(sbatch -W -C gpu run.nsys.daint.slurm 2>&1)
    status1=$?
    grep -q SUCCESS run.nsys.daint.out
    status2=$?
    set -e
    wait
    echo "DONE WAITING ${status1} ${status2}"
    if [ $status1 -ne 0 -o $status2 -ne 0 ] ; then
	if [ ! -z "${BUILD_TAG}" ] ; then
            cleanupFailedJob "${res}" run.nsys.daint.out
	fi
        echo "ERROR: performance run wrapped by nsys not sucessful"
        exit 1
    else
        echo "performance run wrapped by nsys sucessful"
    fi
fi

python examples/standalone/benchmarks/collect_memory_usage_data.py . $githash
