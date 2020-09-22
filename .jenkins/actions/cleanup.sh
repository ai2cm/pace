#!/bin/bash
set -e -x
FV3_TAG="${JOB_NAME}-${BUILD_NUMBER}" make cleanup_remote
