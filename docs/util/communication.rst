.. _communication:

=============
Communication
=============

As mentioned when discussing :ref:`State`, each process or "rank" on a cubed sphere is responsible for a subset of the cubed sphere grid.
In order to operate, the model needs to know how to partition that cubed sphere into parts for each rank, and to be able to communicate data between those ranks.

Partitioning is managed by so-called "Partitioner" objects.
The :py:class:`pace.util.CubedSpherePartitioner` manages the entire cubed sphere, while the :py:class:`pace.util.TilePartitioner` manages one of the six faces of the cube, or a region on one of those faces.
For communication, we similarly have :py:class:`pace.util.CubedSphereCommunicator` and :py:class:`pace.util.TileCommunicator`.
Please see their API documentation for an up-to-date list of current communications routines.

Halo Updates
------------

Let's walk through a detailed example where we create everything we need to perform halo updates on a cubed sphere, to get a feel for the responsibilities of all involved classes. Here we assume that you already know what halo updates are and how data is partitioned in memory-parallel earth system models. If not, there is a (very brief) explanation in the :ref:`State` section, or we recommend searching for information on the "Ghost Cell Pattern" or "Halo Exchange".

TilePartitioner
~~~~~~~~~~~~~~~

First, we create a :py:class:`pace.util.TilePartitioner` object:

.. doctest::

    >>> import pace.util
    >>> partitioner = pace.util.TilePartitioner(layout=(1, 1))

This partitioner will be responsible for partitioning the data on a single face of the cubed sphere into a single tile.
The :py:attr:`pace.util.TilePartitioner.layout` attribute is a tuple of two integers, which specifies how many ranks (processors) to partition the cubed sphere into in the :math:`x` and :math:`y` directions.
For a (1, 1) layout, only one rank will be responsible for each tile face.

.. doctest::

    >>> partitioner.layout
    (1, 1)
    >>> partitioner.total_ranks
    1

The :py:class:`pace.util.TilePartitioner` object is a concrete implementation of the :py:class:`pace.util.Partitioner` abstract base class. Partitioners are responsible for telling us how data on a global model domain is partitioned between ranks, given information about the shapes of the global or local domain and staggering of the data. They do not themselves store this information, meaning the same partitioner can be used to partition data at varying resolutions or with different grid staggering.

Boundary
~~~~~~~~

Within the halo update code, a very important feature of the partitioner is the method :py:meth:`pace.util.Partitioner.boundary`, which returns a :py:class:`pace.util.Boundary` object:

.. doctest::

    >>> boundary = partitioner.boundary(pace.util.EAST, rank=0)
    >>> boundary
    SimpleBoundary(from_rank=0, to_rank=0, n_clockwise_rotations=0, boundary_type=1)

Boundary objects are responsible for describing the boundary between two neighboring ranks, and can tell us what part of a rank's data is on the boundary through its :py:meth:`pace.util.Boundary.send_view` method, and where the neighboring rank's data belongs in the local halo through its :py:meth:`pace.util.Boundary.recv_view` method. As a user you generally will not need to interact with Boundary objects, but they are important to understand if you need to modify or extend the communication code.

.. note::
    The :py:meth:`pace.util.Partitioner.boundary` method will need to be refactored in the future to support non-square layouts.
    The method currently assumes that for a given direction there will be one rank in that direction, but this is not true for tile edges in non-square layouts, and this assumption is not required elsewhere in the code.
    Likely the method should be refactored into one that returns an iterable of all boundaries for a given rank.

Quantity
~~~~~~~~

To see how the boundary and other objects operate, we will need some data to operate on. We use a :py:class:`pace.util.Quantity` object to store the data and all required information about its staggering and halo data:

.. doctest::

    >>> import numpy as np
    >>> quantity = pace.util.Quantity(
    ...     data=np.zeros((6, 6)),
    ...     dims=[pace.util.X_DIM, pace.util.Y_DIM],
    ...     units="m",
    ...     origin=(1, 1),
    ...     extent=(4, 4),
    ... )

This creates a cell-centered Quantity with 8x8x6 data points, and 2 halo points in each direction.
The :py:attr:`pace.util.Quantity.view` attribute provides convenient indexing into the compute domain.
We can see the extent (size) of the compute domain described by the extent of the quantity:

.. doctest::

    >>> quantity.view[:].shape
    (4, 4)
    >>> quantity.extent
    (4, 4)

Given a Quantity, the Boundary object can tell us where the data on the boundary is located:

.. doctest::

    >>> quantity.view[:] = np.arange(4)[None, :] + np.arange(0, 40, 10)[:, None]
    >>> quantity.data[:]
    array([[ 0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  2.,  3.,  0.],
           [ 0., 10., 11., 12., 13.,  0.],
           [ 0., 20., 21., 22., 23.,  0.],
           [ 0., 30., 31., 32., 33.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.]])
    >>> boundary.send_slice(quantity.halo_spec(n_halo=1))
    (slice(4, 5, None), slice(1, 5, None))
    >>> quantity.data[boundary.send_slice(quantity.halo_spec(n_halo=1))]
    array([[30., 31., 32., 33.]])
    >>> boundary.recv_slice(quantity.halo_spec(n_halo=1))
    (slice(5, 6, None), slice(1, 5, None))

.. note::
    Boundary also has some older :py:meth:`pace.util.Boundary.send_view` and :py:meth:`pace.util.Boundary.recv_view` methods which provide similar functionality.
    The original halo update code used these, while a newer pathway that involves building a HaloUpdater and pre-compiling some efficient kernels for data packing uses the "slice" methods.

Comm
~~~~

We've established some objects for containing data and how it is partitioned, but we still need to actually perform the communication.
The low-level object responsible for this is the :py:class:`pace.util.Comm` abstract base class.
This mirrors the comm object provided by the `mpi4py`_ package, which is a thin wrapper over MPI.
There are multiple Comm classes available.
Under normal circumstances when running a parallel model you will want to use a :py:class:`pace.util.MPIComm` object, which is a wrapper around an `mpi4py`_ communicator:

.. doctest::

        >>> import pace.util
        >>> comm = pace.util.MPIComm()
        >>> comm
        <pace.util.mpi.MPIComm object at 0x...>
        >>> comm.Get_rank()
        0
        >>> comm.Get_size()
        1

However this documentation is unit tested, and when it's unit tested it's run on only one rank.
For this reason, many of our tests use the :py:class:`pace.util.NullComm` object, which is a fake communicator that pretends to be an MPI communicator but does not actually perform any communication:

.. doctest::

    >>> comm = pace.util.NullComm(rank=0, total_ranks=6)
    >>> comm
    NullComm(rank=0, total_ranks=6)
    >>> comm.Get_rank()
    0
    >>> comm.Get_size()
    6

This is very useful for testing code that relies on multi-rank communication without actually running a parallel model, at the expense of not being able to rely on or test the numerical values being output.
Keep this in mind below, where we will avoid showing output values after halo updates because the NullComm cannot actually update them.

.. note::
    It is possible to update :py:class:`pace.util.LocalComm` so that it could show a true halo update on one rank, but this is not currently implemented.
    The halo update code currently relies on an assumption that only one boundary exists between any pair of ranks, which is not true for a periodic domain with anything less than a 3x3 tile layout.
    If this does get implemented, this example should be updated (at least for the tile communication case).

TileCommunicator
~~~~~~~~~~~~~~~~

Halo updates and other communication is performed by the :py:class:`pace.util.Communicator` abstract base class.
Code that relies only on the abstract base class should be able to run on any Communicator, including both the :py:class:`TileCommunicator`` which provides a single doubly-periodic tile, or the :py:class:`pace.util.CubedSphereCommunicator`` which provides a cubed sphere decomposition.
We'll start with the single-tile case.

.. doctest::

    >>> comm = pace.util.NullComm(rank=0, total_ranks=9)
    >>> partitioner = pace.util.TilePartitioner(layout=(3, 3))
    >>> tile_communicator = pace.util.TileCommunicator(comm, partitioner)

With all of these objects in place, we can perform an in-place halo update:

.. doctest::

    >>> tile_communicator.halo_update(quantity, n_points=1)

An asynchronous halo update can also be performed:

.. doctest::

    >>> request = tile_communicator.start_halo_update(quantity, n_points=1)
    >>> request.wait()

The communicator provides other communication routines, including scatter/gather and a routine to synchronize interface data computed on both ranks neighboring a boundary.

CubedSphereCommunicator
~~~~~~~~~~~~~~~~~~~~~~~

The :py:class:`pace.util.CubedSphereCommunicator` provides a cubed sphere decomposition of the sphere.
It is used identically to the TileCommunicator, which is by design so that the same code can be used for both decompositions.

.. doctest::

    >>> comm = pace.util.NullComm(rank=0, total_ranks=54)
    >>> partitioner = pace.util.CubedSpherePartitioner(
    ...     pace.util.TilePartitioner(layout=(3, 3))
    ... )
    >>> communicator = pace.util.CubedSphereCommunicator(comm, partitioner)
    >>> communicator.halo_update(quantity, n_points=1)

The :py:class:`pace.util.CubedSpherePartitioner` is a wrapper around a :py:class:`pace.util.TilePartitioner` that provides a cubed sphere decomposition.
Within its implementation, it relies entirely on the TilePartitioner to describe how data is partitioned within any given tile, and only imposes constraints on how tiles are ordered and connected to each other.

.. _mpi4py: https://mpi4py.readthedocs.io/en/stable/
