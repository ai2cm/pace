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

echo "creating the venv"
git submodule update --init --recursive
cd external/daint_venv
./install.sh test_ve
source test_ve/bin/activate
echo "install requirements..."
cd $ROOT_DIR
pip install external/fv3gfs-util/
pip install .

pip list

# set up the experiment data
cp -r $data_path test_data
tar -xf test_data/dat_files.tar.gz -C test_data
cp test_data/*.yml test_data/input.yml

# set the environment
git clone https://github.com/VulcanClimateModeling/buildenv/
cp buildenv/submit.daint.slurm compile.daint.slurm
cp buildenv/submit.daint.slurm run.daint.slurm

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



echo "submitting script to do compilation"
# Adapt batch script to compile the code:
sed -i s/\<NAME\>/standalone/g compile.daint.slurm
sed -i s/\<NTASKS\>/$ranks/g compile.daint.slurm
sed -i s/\<NTASKSPERNODE\>/$ranks/g compile.daint.slurm
sed -i s/\<CPUSPERTASK\>/1/g compile.daint.slurm
sed -i s/--output=\<OUTFILE\>/--hint=nomultithread/g compile.daint.slurm
sed -i s/00:45:00/03:30:00/g compile.daint.slurm
sed -i s/\<G2G\>/export\ CRAY_CUDA_MPS=1/g compile.daint.slurm
sed -i "s#<CMD>#export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:\$PYTHONPATH\nsrun python examples/standalone/runfile/dynamics.py test_data/ 1 $backend $githash#g" compile.daint.slurm

# execute on a gpu node
sbatch -W -C gpu compile.daint.slurm
wait
echo "compilation step finished"
rm *.json

echo "submitting script to do performance run"
# Adapt batch script to run the code:
sed -i s/\<NAME\>/standalone/g run.daint.slurm
sed -i s/\<NTASKS\>/$ranks/g run.daint.slurm
sed -i s/\<NTASKSPERNODE\>/1/g run.daint.slurm
sed -i s/\<CPUSPERTASK\>/$nthreads/g run.daint.slurm
sed -i s/--output=\<OUTFILE\>/--hint=nomultithread/g run.daint.slurm
sed -i s/00:45:00/00:30:00/g run.daint.slurm
sed -i s/cscsci/normal/g run.daint.slurm
sed -i s/\<G2G\>//g run.daint.slurm
sed -i "s#<CMD>#export PYTHONPATH=/project/s1053/install/serialbox2_master/gnu/python:\$PYTHONPATH\nsrun python examples/standalone/runfile/dynamics.py test_data/ $timesteps $backend $githash#g" run.daint.slurm

# execute on a gpu node
sbatch -W -C gpu run.daint.slurm
wait
cp *.json $target_dir
echo "performance run sucessful"
