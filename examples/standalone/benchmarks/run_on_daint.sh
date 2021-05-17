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
SCRIPT=`realpath $0`
SCRIPT_DIR=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
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
        rm -rf .gt_cache_0000*
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
cd $ROOT_DIR
git submodule update --init external/fv3gfs-util external/daint_venv

# set GT4PY version
cd $ROOT_DIR
export GT4PY_VERSION=`cat GT4PY_VERSION.txt`

# set up the virtual environment
echo "creating the venv"
if [ -d ./venv ] ; then rm -rf venv ; fi
cd $ROOT_DIR/external/daint_venv/
if [ -d ./gt4py ] ; then rm -rf gt4py ; fi
./install.sh $ROOT_DIR/venv
cd $ROOT_DIR
source ./venv/bin/activate

# install the local packages
echo "install requirements..."
pip install ./external/fv3gfs-util/
pip install .
pip list

# set the environment
if [ -d ./buildenv ] ; then rm -rf buildenv ; fi
git clone https://github.com/VulcanClimateModeling/buildenv/
cp ./buildenv/submit.daint.slurm run.daint.slurm
if [ "${DO_NSYS_RUN}" == "true" ] ; then
    cp ./buildenv/submit.daint.slurm run.nsys.daint.slurm
fi

if git rev-parse --git-dir > /dev/null 2>&1 ; then
  githash=`git rev-parse HEAD`
else
  githash="notarepo"
fi

echo "Configuration overview:"
echo "    Root dir:          $ROOT_DIR"
echo "    Timesteps:         $timesteps"
echo "    Ranks:             $ranks"
echo "    Backend:           $backend"
echo "    Input data dir:    $data_path"
echo "    Threads per rank:  $NTHREADS"
echo "    GIT hash:          $githash"
echo "    Python arguments:  $py_args"
echo "    Run arguments:     $run_args"
echo "    Extra run in nsys: $DO_NSYS_RUN"

split_path=(${data_path//\// })
experiment=${split_path[-1]}
sample_cache=.gt_cache_000000

if [ ! -d $(pwd)/${sample_cache} ] ; then
    premade_caches=/scratch/snx3000/olifu/jenkins/scratch/store_gt_caches/$experiment/$backend
    if [ -d ${premade_caches}/${sample_cache} ] ; then
	 version_file=${premade_caches}/GT4PY_VERSION.txt
	 if [ -f ${version_file} ]; then
             version=`cat ${version_file}`
	 else
             version=""
	 fi
	 if [ "$version" == "$GT4PY_VERSION" ]; then
	     echo "copying premade GT4Py caches"
             cp -r ${premade_caches}/.gt_cache_0000* .
             find . -name m_\*.py -exec sed -i "s|\/scratch\/snx3000\/olifu\/jenkins_submit\/workspace\/fv3core-cache-setup\/backend\/$backend\/experiment\/$experiment\/slave\/daint_submit|$(pwd)|g" {} +
	 fi
   fi
fi

echo "submitting script to do performance run"
# Adapt batch script to run the code:
sed -i "s/<NAME>/standalone/g" run.daint.slurm
sed -i "s/<NTASKS>/$ranks/g" run.daint.slurm
sed -i "s/<NTASKSPERNODE>/1/g" run.daint.slurm
sed -i "s/<CPUSPERTASK>/$NTHREADS/g" run.daint.slurm
sed -i "s/<OUTFILE>/run.daint.out\n#SBATCH --hint=nomultithread/g" run.daint.slurm
sed -i "s/00:65:00/00:40:00/g" run.daint.slurm
sed -i "s/cscsci/normal/g" run.daint.slurm
sed -i "s/<G2G>/export PYTHONOPTIMIZE=TRUE/g" run.daint.slurm
sed -i "s#<CMD>#export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:\$PYTHONPATH\nsrun python $py_args examples/standalone/runfile/dynamics.py $data_path $timesteps $backend $githash $run_args#g" run.daint.slurm
# execute on a gpu node
set +e
res=$(sbatch -W -C gpu run.daint.slurm 2>&1)
status1=$?
grep -q SUCCESS run.daint.out
status2=$?
set -e
wait
echo "DONE WAITING ${status1} ${status2}"
if [ $status1 -ne 0 -o $status2 -ne 0 ] ; then
    cleanupFailedJob "${res}" run.daint.out
    echo "ERROR: performance run not sucessful"
    exit 1
else
    echo "performance run sucessful"
fi

if [ "${DO_NSYS_RUN}" == "true" ] ; then
    echo "submitting script to do performance run wrapped by nsys"
    # Adapt batch script to run the code:
    sed -i "s/<NAME>/standalone/g" run.nsys.daint.slurm
    sed -i "s/<NTASKS>/$ranks/g" run.nsys.daint.slurm
    sed -i "s/<NTASKSPERNODE>/1/g" run.nsys.daint.slurm
    sed -i "s/<CPUSPERTASK>/$NTHREADS/g" run.nsys.daint.slurm
    sed -i "s/<OUTFILE>/run.nsys.daint.out\n#SBATCH --hint=nomultithread/g" run.nsys.daint.slurm
    sed -i "s/00:45:00/00:40:00/g" run.nsys.daint.slurm
    sed -i "s/cscsci/normal/g" run.nsys.daint.slurm
    sed -i "s#<G2G>#module load nvidia-nsight-systems/2021.1.1.66-6c5c5cb\nexport PYTHONOPTIMIZE=TRUE#g" run.nsys.daint.slurm
    sed -i "s#<CMD>#export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:\$PYTHONPATH\nsrun nsys profile --force-overwrite=true -o %h.%q{SLURM_NODEID}.%q{SLURM_PROCID}.qdstrm --trace=cuda,mpi,nvtx --mpi-impl=mpich python $ROOT_DIR/profiler/external_profiler.py examples/standalone/runfile/dynamics.py $data_path 2 $backend $githash $run_args --disable_json_dump#g" run.nsys.daint.slurm
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
        cleanupFailedJob "${res}" run.nsys.daint.out
        echo "ERROR: performance run wrapped by nsys not sucessful"
        exit 1
    else
        echo "performance run wrapped by nsys sucessful"
    fi
fi

python examples/standalone/benchmarks/collect_memory_usage_data.py . $githash

python examples/standalone/benchmarks/summary.py .
