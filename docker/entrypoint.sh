#!/bin/bash

set -e
pip install -r requirements.txt
pip install -e /fv3gfs-util
pip install -e /fv3core
pip install -e /fv3gfs-physics
exec "$@"
