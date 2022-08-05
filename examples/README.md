This directory serves as a demo of how you can run individual components of pace in a parallel Jupyter notebook.
The component demonstrated is tracer advection, and mirrors the setup of test case 1 in fv3gfs-fortran and corresponds to cosine bell advection.

Note that this proof-of-concept example is only running the tracer advection and none of the other model components.

In addition to having pace built, you will also need the following modules installed:
- `ipyparallel`: (https://pypi.org/project/ipyparallel/)
-  `fv3viz`:
```
git clone https://github.com/ai2cm/fv3net.git
cd fv3net
git checkout
export PYTHONPATH=/fv3net/external/fv3viz
```

The `notebooks` directory contains a helper `functions.py` file and a few notebooks:
- `domain_decomposition_grid_generation.ipynb`: focuses on how to set up a domain and create tools for individual component creation.
- `initial_condition_definition.ipynb`: focuses on how to set up initial conditions on the cubed sphere.
- `tracer_advection.ipynb`: the high-level notebook that has everything set up for cosine bell advection.


<!--
This example builds an MPI-enabled Docker image by adding jupyter notebooks to the pace image and opens it with port forwarding.
You can build this image from within tracer advection directory with:
```
$ docker build -t tracer_advection -f Dockerfile ..
```
and then open the notebook by running:
```
$ docker run -p 8888:8888 tracer_advection
```

You should see output that looks something like:
```
SOMETHING
```

To use the notebook, you need to copy-paste the second URL into your browser. -->
