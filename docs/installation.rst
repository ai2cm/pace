.. meta::
   :robots: noindex, nofollow

.. highlight:: shell

Installation
============

Stable release
--------------

There is no stable release. This is unsupported, pre-alpha software: use at your own risk!

From sources
------------

The sources for fv3gfs-util can be downloaded from the fv3gfs-wrapper `Github repo`_.

You can clone the public repository:

.. code-block:: console

    $ git clone git://github.com/VulcanClimateModeling/fv3gfs-wrapper

Once you have a copy of the source, you can install it interactively with:

.. code-block:: console

    $ pip install -e external/fv3gfs-util

The `-e` flag will set up the directory so that python uses the local folder including
any modifications, instead of copying the sources to an installation directory. This
is very useful for development.

.. _Github repo: https://github.com/VulcanClimateModeling/fv3gfs-wrapper
