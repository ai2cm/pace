# Developing and running Pace components

This directory serves as a demo of how you can develop and run individual components of pace in a parallel Jupyter notebook.

<<<<<<< HEAD
The `notebooks` directory contains a helper `functions.py` file and a few notebooks:
- `domain_decomposition_grid_generation.ipynb`: focuses on how to set up a domain and create tools for individual component creation.
- `initial_condition_definition.ipynb`: focuses on how to set up initial conditions on the cubed sphere.
- ??`building_stencils.ipynb`???: focuses on how to initialize and run stencils (short?)
- `tracer_advection.ipynb`: the high-level notebook that has everything set up for cosine bell advection.
=======
## Getting started

The easiest way of running these demos is using a Docker container. Type `make help` to get help for the different make targets to build, run and stop the container. Once the docker container is running, you can connect to the Jupyter notebook server by copying and pasting the correct address into any browser on your machine.
>>>>>>> 3aa5f970f3b9b4cabc6282c36988ba3fd1b3ba90

The `notebooks` directory contains several notebooks which explain different functionalities required in order to develop and run individual components. The notebooks cover topics such as the generation of a cubed-sphere grid, domain-decomposition, as well as the creation of fields and the generation, compilation and application of individual stencils.

The Jupyter notebooks rely on an MPI-environment and use ipyparallel and mpi4py for setting up the parallel environment. If you are not familiar with these packages, please read up on them before getting started.

If you prefer to run outside of a Docker container, you will need to installe several pre-requisites. The Dockerfile can server as a recipe of what they are and how to install them.
