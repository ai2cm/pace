#!/bin/bash

set -e

pip install -e /external/util -e /external/dsl -e /external/stencils -c constraints.txt
pip install -e /fv3core -c /constraints.txt

exec "$@"
