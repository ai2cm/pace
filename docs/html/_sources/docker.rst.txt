.. highlight:: shell

======
Docker
======

While it is possible to install and build pace bare-metal, we can ensure all system libraries are installed with the correct versions by using a Docker container to test and develop pace.
This requires you have Docker installed (we recommend `Docker Desktop`_ for most users).
You may need to increase memory allocated to Docker in its settings.

Before building the Docker image, you will need to update the git submodules so that any dependencies are cloned and at the correct version:

.. code-block:: console

    $ git submodule update --init --recursive

Then build the `pace` docker image at the top level:

.. code-block:: console

    $ make build

.. _`Docker Desktop`: https://www.docker.com/
