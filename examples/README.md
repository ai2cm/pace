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
