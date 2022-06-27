#!/bin/bash -f

# This is the main script used to trigger Jenkins actions.
# The idea of this script is to keep the amount of code in the "Execute shell" field small
#
# Example syntax:
# .jenkins/jenkins.sh test
#
# Other actions such as test/build/deploy can be defined.

### Some environment variables available from Jenkins
### Note: for a complete list see https://jenkins.ginko.ch/env-vars.html
# slave              The name of the build worker (daint, kesch, ...).
# BUILD_NUMBER       The current build number, such as "153".
# BUILD_ID           The current build id, such as "2005-08-22_23-59-59" (YYYY-MM-DD_hh-mm-ss).
# BUILD_DISPLAY_NAME The display name of the current build, something like "#153" by default.
# NODE_NAME          Name of the worker if the build is on a worker, or "master" if run on main worker.
# NODE_LABELS        Whitespace-separated list of labels that the node is assigned.
# JENKINS_HOME       The absolute path of the data storage directory assigned on the master node.
# JENKINS_URL        Full URL of Jenkins, like http://server:port/jenkins/
# BUILD_URL          Full URL of this build, like http://server:port/jenkins/job/foo/15/
# JOB_URL            Full URL of this job, like http://server:port/jenkins/job/foo/

set -x +e

JENKINS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BUILDENV_DIR=$JENKINS_DIR/../../buildenv

# some global variables
action="$1"
optarg="$2"

# Timeout after this many minutes
minutes=45

# get latest version of buildenv
git submodule update --init

# setup module environment and default queue
. ${BUILDENV_DIR}/machineEnvironment.sh

# load machine dependent environment
. ${BUILDENV_DIR}/env.${host}.sh

# load scheduler tools (provides run_command)
. ${BUILDENV_DIR}/schedulerTools.sh

set -e

# check if action script exists
script="${JENKINS_DIR}/actions/${action}.sh"
test -f "${script}" || exitError 1301 ${LINENO} "cannot find script ${script}"

# set up virtual env
python3 --version
python3 -m venv venv
. ./venv/bin/activate
pip3 install --upgrade pip setuptools wheel
pip3 install -r $JENKINS_DIR/../requirements.txt -c $JENKINS_DIR/../../constraints.txt
pip3 install -e ${JENKINS_DIR}/../external/gt4py -c $JENKINS_DIR/../../constraints.txt
pip3 install -e ${JENKINS_DIR}/../ -c $JENKINS_DIR/../../constraints.txt

set +e

if [ "${target}" == "cpu" ] ; then
  scheduler = "none"
fi

echo "I am running on host ${host} with scheduler ${scheduler}."
run_command "${script} ${optarg}" "UtilAction${action}" $minutes

if [ $? -ne 0 ] ; then
  exitError 1510 ${LINENO} "problem while executing script ${script}"
fi
echo "### ACTION ${action} SUCCESSFUL"
