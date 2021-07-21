#!/bin/bash

set -e
pip install -e /fv3gfs-physics/
exec "$@"