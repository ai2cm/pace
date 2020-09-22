#!/bin/bash
set -e -x
pip3 install black==19.10b0 flake8==3.7.8
export PATH=/home/jenkins/.local/bin:${PATH}
make lint
echo $(date) > aggregate
