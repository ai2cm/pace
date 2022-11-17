# Developing and running Pace components

This directory serves as a demo of how you can develop and run individual components of pace in a parallel Jupyter notebook.

## Getting started

The easiest way of running these demos is using a Docker container. If you do not already have Docker installed, it can be downloaded directly from [Docker](https://www.docker.com/). Once Docker is set up, you can build and run the container:

```
make build
make run
```

Once the docker container is running, you can connect to the Jupyter notebook server by copying and pasting the URLs into any browser on your machine. An example output is shown below:

```
Serving notebooks from local directory: /notebooks
Jupyter Server 1.19.1 is running at:
http://0d7f58225e26:8888/lab?token=e23ef7f3a334d97d8a74d499839b0dcfa91c879f7effa968
or http://127.0.0.1:8888/lab?token=e23ef7f3a334d97d8a74d499839b0dcfa91c879f7effa968
Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    To access the server, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/jpserver-1-open.html
    Or copy and paste one of these URLs:
        http://0d7f58225e26:8888/lab?token=e23ef7f3a334d97d8a74d499839b0dcfa91c879f7effa968
     or http://127.0.0.1:8888/lab?token=e23ef7f3a334d97d8a74d499839b0dcfa91c879f7effa968
```

Use the last URL `http://127.0.0.1:8888/lab?token=XXX` to connect to the Jupyter notebook server.

If you would like to retain the changes made to the notebooks, you can mount the local notebooks folder into the container by running `make dev` instead of `make run`.

Type `make help` to see more make targets on building and running the container.

The `notebooks` directory contains several notebooks which explain different functionalities required in order to develop and run individual components. The notebooks cover topics such as the generation of a cubed-sphere grid, domain-decomposition, as well as the creation of fields and the generation, compilation and application of individual stencils.

The Jupyter notebooks rely on an MPI-environment and use ipyparallel and mpi4py for setting up the parallel environment. If you are not familiar with these packages, please read up on them before getting started.

If you prefer to run outside of a Docker container, you will need to install several pre-requisites. The Dockerfile can serve as a recipe of what they are and how to install them.

## Building an environment to run these examples on the GFDL Post Processing and Analysis Cluster (PP/AN)

Within the `build_scripts` directory are a couple scripts relevant for setting up an environment to run these notebook examples on the [GFDL Post Processing and Analysis Cluster (PP/AN)](https://www.noaa.gov/organization/information-technology/ppan).  To create an environment, log in to either the `jhan` or `jhanbigmem` node on the analysis cluster clone this repository, and then navigate to the `build_scripts` directory:

```
$ ssh analysis
$ git clone --recursive https://github.com/ai2cm/pace.git
$ cd pace/examples/build_scripts
```

From there, run the installation script.  This script takes two arguments.  The first is to a directory you would like to install all the software into, and the second is a name you would like to give the conda environment the script creates.  Since conda environments can take up a reasonable amount of space, it is recommended that you install things in your `/work` directory rather than your `/home` directory:

```
$ bash build_ppan.sh /work/$USER/pace-software pace
```

This will take a few minutes to complete, but assuming it finishes succesfully you should then be able to activate the environment and run the example notebooks.  To do so, first source the `activate_ppan.sh` script.  This loads some modules, activates the appropriate conda environment, and sets some environment variables necessary for running the examples.  You will then be able to start JupyterLab following [GFDL's recommended approach](https://wiki.gfdl.noaa.gov/index.php/Python_at_GFDL#Using_Jupyter_Hubs_or_Notebooks_on_GFDL_Workstations_and_PPAN):

```
$ source activate_ppan.sh /work/$USER/pace-software pace
$ cd ..
$ jhp launch lab
```

It will take a minute or so for the server to start up.  If you are on GFDL's network or have [configured your proxy settings appropriately](https://wiki.gfdl.noaa.gov/index.php/Creating_a_GFDL_SSH_Tunnel), you will then be able to navigate to the URL produced by the command in your local browser.  This will take you to the JupyterLab interface, where you can open and run the example notebooks.  Note that it can take some time to initially connect to the notebook kernel, so be patient.
