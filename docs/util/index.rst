=========
pace-util
=========

pace-util is a toolkit for building Python weather and climate models.
Its features can seem disjoint, which is by design - you can choose which functionality you want to use and leave the rest.
It is currently used to contain pure Python utilities shared by `fv3gfs-wrapper`_ and `fv3core`_.
As the number of features increases, we may move its functionality into separate packages to reduce the dependency stack.

Some broad categories of features are:

- :py:class:`pace.util.Quantity`, the data type used by pace-util described in the section on :ref:`State`
- :ref:`Communication` objects used for MPI
- Utility functions useful in weather and climate models, described in :ref:`Utilities`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   state
   communication
   utilities
   api
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _fv3gfs-wrapper: https://github.com/VulcanClimateModeling/fv3gfs-wrapper
.. _fv3core: https://github.com/VulcanClimateModeling/fv3core
