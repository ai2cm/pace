import sys
from dataclasses import fields

import fv3core.initialization.baroclinic as baroclinic_init
import pace.util
from fv3core import DynamicalCore
from fv3core.initialization.dycore_state import DycoreState
from fv3core.testing.translate_fvdynamics import init_dycore_state_from_serialized_data
from fv3gfs.physics import PhysicsState
from fv3gfs.physics.stencils.physics import Physics
from pace.stencils.testing.grid import DampingCoefficients, DriverGridData, GridData
from pace.stencils.update_atmos_state import UpdateAtmosphereState
from pace.util.constants import N_HALO_DEFAULT
from pace.util.grid import MetricTerms


class Driver:
    def __init__(
        self,
        namelist,
        comm,
        backend,
        dycore_init_mode="serialized_data",
        rebuild=False,
        validate_args=True,
        data_dir=None,
    ):
        self._data_dir = data_dir
        self.namelist = namelist
        self.backend = backend
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(self.namelist.layout)
        )
        self.comm = pace.util.CubedSphereCommunicator(comm, partitioner)
        self.quantity_factory, self.stencil_factory = self.setup_factories(
            rebuild, validate_args
        )
        self.metric_terms, self.grid_data = self.compute_grid()
        self.dycore_state = self.initialize_dycore_state(dycore_init_mode)

        self.dycore = DynamicalCore(
            comm=self.comm,
            grid_data=self.grid_data,
            stencil_factory=self.stencil_factory,
            damping_coefficients=DampingCoefficients.new_from_metric_terms(
                self.metric_terms
            ),
            config=self.namelist.dynamical_core,
            phis=self.dycore_state.phis,
        )
        if not self.namelist.dycore_only:
            self.physics = Physics(
                stencil_factory=self.stencil_factory,
                grid_data=self.grid_data,
                namelist=self.namelist,
            )
            self.physics_state = self.init_physics_from_dycore()
            self.state_updater = UpdateAtmosphereState(
                stencil_factory=self.stencil_factory,
                grid_data=self.grid_data,
                namelist=self.namelist,
                comm=self.comm,
                grid_info=DriverGridData.new_from_metric_terms(self.metric_terms),
                quantity_factory=self.quantity_factory,
            )

    def setup_factories(self, rebuild, validate_args):
        stencil_config = pace.dsl.stencil.StencilConfig(
            backend=self.backend,
            rebuild=rebuild,
            validate_args=validate_args,
        )
        sizer = pace.util.SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=N_HALO_DEFAULT,
            extra_dim_lengths={},
            layout=self.namelist.layout,
            tile_partitioner=self.comm.partitioner.tile,
            tile_rank=self.comm.tile.rank,
        )

        grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
            sizer=sizer, cube=self.comm
        )
        quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer, backend=self.backend
        )
        stencil_factory = pace.dsl.stencil.StencilFactory(
            config=stencil_config,
            grid_indexing=grid_indexing,
        )
        return quantity_factory, stencil_factory

    def compute_grid(self):
        metric_terms = MetricTerms(
            quantity_factory=self.quantity_factory, communicator=self.comm
        )
        grid_data = GridData.new_from_metric_terms(metric_terms)
        return metric_terms, grid_data

    def initialize_dycore_state(self, dycore_init_mode):
        if dycore_init_mode == "serialized_data":
            sys.path.append("/usr/local/serialbox/python/")
            import serialbox

            serializer = serialbox.Serializer(
                serialbox.OpenModeKind.Read,
                self._data_dir,
                "Generator_rank" + str(self.comm.rank),
            )
            dycore_state = init_dycore_state_from_serialized_data(
                serializer=serializer,
                rank=self.comm.rank,
                backend=self.backend,
                namelist=self.namelist.dynamical_core,
                quantity_factory=self.quantity_factory,
                stencil_factory=self.stencil_factory,
            )

        elif dycore_init_mode == "baroclinic":
            # create an initial state from the Jablonowski & Williamson Baroclinic
            # test case perturbation. JRMS2006

            dycore_state = baroclinic_init.init_baroclinic_state(
                self.metric_terms,
                adiabatic=self.namelist.adiabatic,
                hydrostatic=self.namelist.hydrostatic,
                moist_phys=self.namelist.moist_phys,
                comm=self.comm,
            )
        else:
            raise ValueError(
                "dycore_init_mode is "
                + dycore_init_mode
                + ", which is not one of the supported options"
            )
        return dycore_state

    def init_physics_from_dycore(self):
        initial_storages = {}
        dycore_fields = fields(DycoreState)
        for field in fields(PhysicsState):
            if field.metadata["full_model_var"]:
                initial_storages[field.name] = getattr(self.dycore_state, field.name)
            else:
                initial_storages[field.name] = self.quantity_factory.zeros(
                    [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
                    field.metadata["units"],
                    dtype=float,
                )
        return PhysicsState(**initial_storages, quantity_factory=self.quantity_factory)

    def step_dynamics(self, do_adiabatic_init, timestep):
        self.dycore.step_dynamics(
            state=self.dycore_state,
            conserve_total_energy=self.namelist.consv_te,
            n_split=self.namelist.n_split,
            do_adiabatic_init=do_adiabatic_init,
            timestep=timestep,
        )

    def step_physics(self):
        if not self.namelist.dycore_only:
            self.physics(self.physics_state)
            self.state_updater(self.dycore_state, self.physics_state)
