#!/bin/bash
set -e -x
JENKINS_TAG="${JOB_NAME}-${BUILD_NUMBER}" make cleanup_remote
