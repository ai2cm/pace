#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

docker run -v ${SCRIPT_DIR}/../../:/pace -w /pace python:3.8 bash -c "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-dev libopenmpi3 libopenmpi-dev && cd /pace/driver/examples && /pace/driver/examples/create_venv.sh && . venv/bin/activate && mpirun -n 6 --allow-run-as-root --mca btl_vader_single_copy_mechanism none python3 -m pace.driver.run /pace/driver/examples/configs/baroclinic_c12.yaml"
