.. highlight:: shell

============
Installation
============

Stable release
--------------

There is no stable release. This is alpha research software: use at your own risk!

From sources
------------

The sources for pace-util can be downloaded from the `Github repo`_.
To develop pace-util, you can clone the public repository:

.. code-block:: console

    $ git clone git://github.com/ai2cm/pace

Once you have a copy of the source, you can install it in develop mode from the Pace top level directory with:

.. code-block:: console

    $ pip install -r ./pace-util/requirements.txt -c ./constraints.txt -e pace-util

The `-e` flag will set up the directory so that python uses the local folder including
any modifications, instead of copying the sources to an installation directory. This
is very useful for development. The `-r requirements.txt` will install extra packages
useful for test, lint & other development requirements.

The `-c ./constraints.txt` is optional, but will ensure the package versions you use are ones we have tested against.

.. _Github repo: https://github.com/VulcanClimateModeling/pace-util
