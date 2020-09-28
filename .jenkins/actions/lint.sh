#!/bin/bash
set -e -x
pip3 install -r requirements.txt
PIP_BIN_DIR=/home/jenkins/.local/bin
${PIP_BIN_DIR}/pre-commit run --all-files
echo $(date) > aggregate
