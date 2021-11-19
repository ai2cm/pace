#!/usr/bin/env bash

set -e

echo $(md5sum $@ | md5sum | awk '{print $1;}')
