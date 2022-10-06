# Developing and running Pace components

This directory serves as a demo of how you can develop and run individual components of pace in a parallel Jupyter notebook.

## Getting started

The easiest way of running these demos is using a Docker container. If you do not already have Docker installed, it can be downloaded directly from [Docker](https://www.docker.com/). Once Docker is set up, you can build and run the container:

```
make build
make run
```

Once the docker container is running, you can connect to the Jupyter notebook server by copying and pasting the URLs into any browser on your machine. If you would like to retain the changes made to the notebooks, you can mount the local notebooks folder to the container. Instead of `make run`, do `make dev`.

Type `make help` to see more make targets on building and running the container.

The `notebooks` directory contains several notebooks which explain different functionalities required in order to develop and run individual components. The notebooks cover topics such as the generation of a cubed-sphere grid, domain-decomposition, as well as the creation of fields and the generation, compilation and application of individual stencils.

The Jupyter notebooks rely on an MPI-environment and use ipyparallel and mpi4py for setting up the parallel environment. If you are not familiar with these packages, please read up on them before getting started.

If you prefer to run outside of a Docker container, you will need to install several pre-requisites. The Dockerfile can serve as a recipe of what they are and how to install them.
