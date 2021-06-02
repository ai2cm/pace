#!/bin/sh

# Template for running NCU and getting a file to open in `ncu-ui`
#  * nsight-compute/2021.1.0/ncu: [TO ADAPT] local install of ncu
#  * -f -o results.nvprof: result filename for the profiler file (-f force overwrites the files if it exists)
#  * --target-processes all: make sure all process gets measured, needed for python
#  * --import-source on: tag the soruce so they are available in `ncu-ui`
#  * /home/floriand/venv/vcm_1_0/bin/python3: [TO ADAPT] local venv installed python3
#  *repro_compute_x_flux_2021-05-10_14-30-55/m_compute_x_flux__gtcuda_1bdfdf5c50/repro.py: [TO ADAPT] script launching the stencil from a captured reproducer

sudo /opt/nvidia/nsight-compute/2021.1.0/ncu -f -o results.nvprof \
    --target-processes all \
    --import-source on \
    /home/floriand/venv/vcm_1_0/bin/python3 \
    repro_compute_x_flux_2021-05-10_14-30-55/m_compute_x_flux__gtcuda_1bdfdf5c50/repro.py
