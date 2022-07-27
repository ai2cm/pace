from dataclasses import fields
from datetime import timedelta

import numpy as np

import fv3gfs.physics
import pace.dsl
import pace.util
import pace.util.grid
from pace.dsl.stencil_config import CompilationConfig
from pace.stencils.testing import assert_same_temporaries, copy_temporaries


def setup_physics():
    backend = "numpy"
    layout = (1, 1)
    physics_config = fv3gfs.physics.PhysicsConfig(
        dt_atmos=225, hydrostatic=False, npx=13, npy=13, npz=79, nwat=6, do_qa=True
    )
    mpi_comm = pace.util.NullComm(
        rank=0, total_ranks=6 * layout[0] * layout[1], fill_value=0.0
    )
    partitioner = pace.util.CubedSpherePartitioner(pace.util.TilePartitioner(layout))
    communicator = pace.util.CubedSphereCommunicator(mpi_comm, partitioner)
    sizer = pace.util.SubtileGridSizer.from_tile_params(
        nx_tile=physics_config.npx - 1,
        ny_tile=physics_config.npy - 1,
        nz=physics_config.npz,
        n_halo=3,
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )
    grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
        sizer=sizer, cube=communicator
    )
    quantity_factory = pace.util.QuantityFactory.from_backend(
        sizer=sizer, backend=backend
    )
    dace_config = pace.dsl.DaceConfig(
        communicator=communicator,
        backend=backend,
        orchestration=pace.dsl.DaCeOrchestration.Python,
    )
    stencil_config = pace.dsl.stencil.StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            rebuild=False,
            validate_args=True,
        ),
        dace_config=dace_config,
    )
    stencil_factory = pace.dsl.stencil.StencilFactory(
        config=stencil_config,
        grid_indexing=grid_indexing,
    )
    metric_terms = pace.util.grid.MetricTerms(
        quantity_factory=quantity_factory,
        communicator=communicator,
    )
    grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)
    physics = fv3gfs.physics.Physics(
        stencil_factory, grid_data, physics_config, active_packages=["microphysics"]
    )
    physics_state = fv3gfs.physics.PhysicsState.init_zeros(
        quantity_factory, active_packages=["microphysics"]
    )
    random = np.random.RandomState(0)
    for field in fields(fv3gfs.physics.PhysicsState):
        array = getattr(physics_state, field.name)
        # check that it's a storage this way, because Field is not a class
        if hasattr(array, "data"):
            array.data[:] = random.uniform(-1, 1, size=array.data.shape)
    return physics, physics_state


# TODO: The orchestrated code pushed us to make the dycore stateful for halo
# exchange. This needs to be reactivated after halo exchange are reverted to
# not being stateful.
def test_call_on_same_state_same_dycore_produces_same_temporaries():
    """
    Assuming the precursor test passes, this test indicates whether
    the dycore retains and re-uses internal state on subsequent calls.
    If it does not, then subsequent calls on identical input should
    produce identical results.
    """
    physics, state_1 = setup_physics()
    _, state_2 = setup_physics()

    # state_1 and state_2 are identical, if the dycore is stateless then they
    # should produce identical dycore final states when used to call
    physics(state_1, timestep=timedelta(minutes=5).total_seconds())
    first_temporaries = copy_temporaries(physics, max_depth=10)
    assert len(first_temporaries) > 0
    physics(state_2, timestep=timedelta(minutes=5).total_seconds())
    second_temporaries = copy_temporaries(physics, max_depth=10)
    assert_same_temporaries(second_temporaries, first_temporaries)
