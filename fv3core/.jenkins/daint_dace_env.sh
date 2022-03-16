#!/bin/bash
set -e -x
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
