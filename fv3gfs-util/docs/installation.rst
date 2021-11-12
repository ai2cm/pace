.. highlight:: shell

Installation
============

Stable release
--------------

There is no stable release. This is alpha research software: use at your own risk!

From sources
------------

fv3gfs-util can be installed from Github using:

.. code-block:: console

    $ pip install git+https://github.com/VulcanClimateModeling/fv3gfs-util.git

The sources for fv3gfs-util can be downloaded from the `Github repo`_. To develop fv3gfs-util, you can clone the public repository:

.. code-block:: console

    $ git clone git://github.com/VulcanClimateModeling/fv3gfs-util

Once you have a copy of the source, you can install it in develop mode with:

.. code-block:: console

    $ pip install -r ./fv3gfs-util/requirements.txt -c ./fv3gfs-util/constraints.txt -e fv3gfs-util

The `-e` flag will set up the directory so that python uses the local folder including
any modifications, instead of copying the sources to an installation directory. This
is very useful for development. The `-r requirements.txt` will install extra packages
useful for test, lint & other development requirements. The `-r requirements_gpu.txt`
will install required packages for GPU compatibility.
The `-c constraints.txt` is optional, but will ensure the package versions you use are ones we have tested against.

.. _Github repo: https://github.com/VulcanClimateModeling/fv3gfs-util
