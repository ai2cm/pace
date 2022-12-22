============
pace-physics
============

pace-physics includes the Python implementation of GFS physics built using the GT4Py domain-specific language.
Currently, only GFDL cloud microphysics is integrated into Pace.
Additional physics schemes (NOAH land surface, GFS sea ice, scale-aware mass-flux shallow convection, hybrid eddy-diffusivity mass-flux PBL and free atmospheric turbulence, and rapid radiative transfer model) have been ported indendepently and are available in the `physics-standalone`_ repository.
Additional work is required to integrate these schemes.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   state
   api

Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _physics-standalone: https://github.com/ai2cm/physics_standalone
