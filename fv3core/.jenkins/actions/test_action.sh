#!/bin/bash
set -e -x
echo "${JOB_NAME}-${BUILD_NUMBER}"
echo `pip list`
echo `which python`
