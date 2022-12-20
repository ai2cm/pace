.. _state:

=====
State
=====

Containers
------------
Variables used in physics are packedged using a container type called :py:class:`pace.physics.PhysicsState`.
This contains variables copied from the dynamical core for calculating physics tendencies.
It also contains sub-container for the individual physics schemes. Currently, it only contains :py:class:`pace.physics.MicrophysicsState`.

An example to initialize a PhysicsState and MicrophysicsState is shown below:

.. doctest::

    >>> from pace.util import (
        ...  CubedSphereCommunicator,
        ...  CubedSpherePartitioner,
        ...  Quantity,
        ...  QuantityFactory,
        ...  SubtileGridSizer,
        ...  TilePartitioner,
        ...  NullComm,
        ... )
    >>> from pace.physics import PhysicsState
    >>> layout = (1, 1)
    >>> partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    >>> communicator = CubedSphereCommunicator(NullComm(rank=0, total_ranks=6), partitioner)
    >>> sizer = SubtileGridSizer.from_tile_params(
    ...    nx_tile=12,
    ...    ny_tile=12,
    ...    nz=79,
    ...    n_halo=3,
    ...    extra_dim_lengths={},
    ...    layout=layout,
    ...    tile_partitioner=partitioner.tile,
    ...    tile_rank=communicator.tile.rank,
    ... )

    >>> quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend="numpy")
    >>> physics_state = PhysicsState.init_zeros(
    ...  quantity_factory=quantity_factory, active_packages=["microphysics"]
    ... )
    >>> microphysics_state = physics_state.microphysics
