This directory serves as a demo of how you can run individual components of pace in a parallel Jupyter notebook. The component demonstrated is tracer advection, and mirrors the setup of test case 1 in fv3gfs-fortran and corresponds to cosine bell advection. 

Note that this proof-of-concept example is only running the tracer advection and none of the other model components.

This example builds an MPI-enabled Docker image by adding jupyter notebooks to the pace image and opens it with port forwarding. You can build this image with:
```
$ make build
```
and then open the notebook by running:
```
$ make run
```

You should see output that looks something like:
```
SOMETHING
```

To use the notebook, you need to copy-paste the second URL into your browser.
