#!/usr/bin/env bash

backend=$1
experiment=$2

EXPERIMENT=$experiment TARGET=driver make get_test_data

.jenkins/initialize_driver.py test_data/8.1.0/$experiment/driver $backend
