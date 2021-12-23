#!/bin/bash
if [[ "${NODE_NAME}" == *"daint"* ]] ; then source ~/.bashrc ; fi
set -e
export LONG_EXECUTION=1
.jenkins/jenkins.sh run_physics_all_tests ${backend} ${experiment}
.jenkins/generate_caches.sh ${backend} ${experiment}
