# Build Environment
`buildenv` is a repository that tries to "abstract" the specific environments on different systems in order to simplify writing bash scripts on these systems and make results reproducible (for exmple for a CI system).

## Goals
- Provide a set of standard environmental variables in order to avoid a proliferation of system-specific conditionals in testing, building and deployment plans.
- Encourage different projects or CI plans to use the same environment for testing and building, in order to avoid conflicts further down the pipeline.

## Components
It currently contains the following components:
- Defines a set of standardized variables (host, scheduler, queue, nthreads, mpilaunch, ...) in `machineEnvironment.sh`
- Defines a machine-specific environment (module load ..., export LD_LIBRARY_PATH=...) in `env.${host}.sh` scripts that can be sourced.
- Provides a set of tools for working with/without a scheduler (e.g. SLURM) in `schedulerTools.sh` and host-specific template job submission scripts in `submit.${host}.${scheduler}`.
- Provides a set of tools for querying the git environment (if in a git repo) in `gitTools.sh`.
- Provides a set of tools for working with a system that uses modules in `moduleTools.sh`.
- Provides functionality to issue errors and warnings in `machineEnvironment.sh`.

## Usage
The functionality of `buildenv` is typically accessed from a shell script as in the bash script below. Depending on the use-case, this repository can simply be cloned or sub-moduled into a working environment.

```bash
#!/bin/bash

envloc="."

# get latest version of buildenv
if [ -d ${envloc}/env ] ; then
    cd ${envloc}; git pull; cd -
else
    git clone git@github.com:VulcanClimateModeling/buildenv.git ${envloc}/env
fi

# setup module environment and default queue
. ${envloc}/env/machineEnvironment.sh

# load machine dependent environment
. ${envloc}/env/env.${host}.sh

# load scheduler tools (provides run_command)
. ${envloc}/env/schedulerTools.sh

# rest of script (which uses buildenv functionality)
echo "I am running on host ${host} with scheduler ${scheduler}."
run_command "echo 'This submits a job on systems which have a batch system'"
...

```

## Example
An example of how `buildenv` is being used in a Jenkins CI plan can be found [here](https://github.com/VulcanClimateModeling/fv3gfs-wrapper/tree/master/.jenkins).

## Committing to this repository
Changes to this repository can potentially have dangerous side-effects in all places that use `buildenv` and should be done with care & consideration. In general, it is good practice to do the following:
- [ ] Open a PR and have somebody from the team familiar with `buildenv` and how it is used review it.
- [ ] Notify team that you will be making a change to the `buildenv` to hold back with other PRs or pushes.
- [ ] Merge PR (never on a Friday! never just before or during CSCS maintenance!).
- [ ] Manually force CI plans to rebuild in order to make sure everything is ok.
- [ ] Send a message to the team to notify everybody about the change and which plans you have manually triggered on CI.
- [ ] Verify everything is working correctly.

## Supported systems
- daint: Piz Daint at CSCS in Lugano
- gce: Google Cloud instances instantiated using the [jenkins-agent](https://console.cloud.google.com/compute/imagesDetail/projects/vcm-ml/global/images/jenkins-agent-1593727237?project=vcm-ml&authuser=1&folder&organizationId) image.
- circleci: Circle CI instances
- papaya: Macintosh client
