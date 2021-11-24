#!/bin/bash

set -e
pip install -r requirements.txt -c constraints.txt
pip install -e /fv3gfs-util -c constraints.txt
pip install -e /fv3core -c constraints.txt
pip install -e /fv3gfs-physics -c constraints.txt
pip install -e /stencils -c constraints.txt
exec "$@"
