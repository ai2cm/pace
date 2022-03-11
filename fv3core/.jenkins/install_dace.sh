#!/bin/bash
set -e -x
orchestration=$1
if [ -d "gt4py" ]; then rm -rf gt4py; fi
git clone https://github.com/ai2cm/gt4py.git
cd gt4py
git checkout 47021270f067aff314a0344637d090657f1460d8
cd ../
pip install -e ./gt4py

if [ -d "dace" ]; then rm -rf dace; fi
git clone https://github.com/spcl/dace.git
cd dace
git checkout 56c42b21b72b043408b725dd85d6a47ba62f451b
git submodule update --init
cd ../
pip install -e ./dace

set +e -x
# TODO: this somehow does not work with 9 or 10
module switch gcc gcc/8.3.0

set -e -x
if [ "$orchestration"  == "dace" ] ; then
    echo "setting dace ochestration specific flags"
    export FV3_DACEMODE=True
    export DACE_execution_general_check_args=0
    export DACE_frontend_dont_fuse_callbacks=1
    export DACE_compiler_cpu_openmp_sections=0
    export DACE_compiler_cuda_max_concurrent_streams=-1
    export DACE_frontend_unroll_threshold=0
    export DACE_compiler_unique_functions=none
fi
