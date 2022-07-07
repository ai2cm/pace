=============
Communication
=============

As mentioned when discussing :ref:`State`, each process or "rank" on a cubed sphere is responsible
for a subset of the cubed sphere grid. In order to operate, the model needs to know
how to partition that cubed sphere into parts for each rank, and to be able to
communicate data between those ranks.

Partitioning is managed by so-called "Partitioner" objects. The
:py:class:`pace.util.CubedSpherePartitioner` manages the entire cubed sphere, while the
:py:class:`pace.util.TilePartitioner` manages one of the six faces of the cube, or a
region on one of those faces. For communication, we similarly have
:py:class:`pace.util.CubedSphereCommunicator` and :py:class:`pace.util.TileCommunicator`.
Please see their API documentation for an up-to-date list of current communications
routines.
