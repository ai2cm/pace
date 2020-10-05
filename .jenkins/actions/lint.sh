#!/bin/bash
set -e -x
env_name=venv-${BUILD_NUMBER:-0}
python3 -m venv ${env_name}
. ${env_name}/bin/activate
pip install -r requirements.txt
pre-commit run --all-files
deactivate
echo $(date) > aggregate
