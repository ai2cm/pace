#!/bin/bash

## Arguments:
# $1: number of timesteps to run
# $2: number of ranks to execute with (ensure that this is compatible with fv3core)
# $3: backend to use in gt4py
# $4: target directory to store the output in
# $5: path to the data directory that should be run
#############################################
# Example syntax:
# ./run_on_daint.sh 60 6 gtx86

exitError()
{
    echo "ERROR $1: $3" 1>&2
    echo "ERROR     LOCATION=$0" 1>&2
    echo "ERROR     LINE=$2" 1>&2
    exit $1
}

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`
ROOT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPTPATH")")")"

# check sanity of environment
test -n "$1" || exitError 1001 ${LINENO} "must pass a number of timesteps"
timesteps="$1"
test -n "$2" || exitError 1002 ${LINENO} "must pass a number of ranks"
ranks="$2"

backend="$3"
if [ -z "$3" ]
  then
    backend="numpy"
fi

target_dir="$4"
if [ -z "$4" ]
  then
    target_dir="$ROOT_DIR"
fi

data_path="$5"
if [ -z "$5" ]
  then
    data_path="/project/s1053/fv3core_serialized_test_data/7.0.0/c12_6ranks_standard/"
fi


# set up the virtual environment
cd $ROOT_DIR
rm -rf vcm_1.0

echo "copying in the venv..."
cp -r /project/s1053/install/venv/vcm_1.0/ .
git submodule update --init --recursive
echo "install requirements..."
vcm_1.0/bin/python -m pip install external/fv3gfs-util/
vcm_1.0/bin/python -m pip install .

# set up the experiment data
cp -r $data_path test_data
tar -xf test_data/dat_files.tar.gz -C test_data
cp test_data/*.yml test_data/input.yml

# set the environment
git clone https://github.com/VulcanClimateModeling/buildenv/
source buildenv/machineEnvironment.sh
source buildenv/env.${host}.sh
nthreads=12

echo "Configuration overview:"
echo "    Timesteps:        $timesteps"
echo "    Ranks:            $ranks"
echo "    Threads per rank: $nthreads"
echo "    Input data dir:   $data_path"
echo "    Output dir:       $target_dir"
echo "    Slurm output dir: $ROOT_DIR"

if git rev-parse --git-dir > /dev/null 2>&1; then
  githash=`git rev-parse HEAD`
else
  githash="notarepo"
fi

# Adapt batch script:
cp buildenv/submit.daint.slurm .
sed -i s/\<NAME\>/standalone/g submit.daint.slurm
sed -i s/\<NTASKS\>/$ranks/g submit.daint.slurm
sed -i s/\<NTASKSPERNODE\>/1/g submit.daint.slurm
sed -i s/\<CPUSPERTASK\>/$nthreads/g submit.daint.slurm
sed -i s/--output=\<OUTFILE\>/--hint=nomultithread/g submit.daint.slurm
sed -i s/00:45:00/03:30:00/g submit.daint.slurm
sed -i s/cscsci/normal/g submit.daint.slurm
sed -i s/\<G2G\>//g submit.daint.slurm
sed -i "s#<CMD>#export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:\$PYTHONPATH\nsrun vcm_1.0/bin/python examples/standalone/runfile/dynamics.py test_data/ $timesteps $backend $githash#g" submit.daint.slurm

# execute on a gpu node
sbatch -W -C gpu submit.daint.slurm
wait
cp *.json $target_dir
