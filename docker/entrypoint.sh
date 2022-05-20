#!/bin/bash

set -e
./install.sh
exec "$@"
