import fv3core
import pace.util


# getting namelist from file:
# namelist = pace.util.Namelist.from_f90nml(
#    f90nml.read(os.path.join(data_path, "input.nml"))
# )


class GeosDycoreWrapper:
    def __init__(self, namelist, comm, backend):
        self.namelist = namelist

        self.dycore_config = fv3core.DynamicalCoreConfig.from_namelist(self.namelist)

        self.layout = self.dycore_config.layout
        self.partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(self.layout)
        )
        self.communicator = pace.util.CubedSphereCommunicator(comm, self.partitioner)

        self.sizer = pace.util.SubtileGridSizer.from_namelist(
            self.namelist, self.partitioner.tile, self.communicator.tile.rank
        )
        self.quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer=self.sizer, backend=backend
        )

        # set up the metric terms and grid data
        self.metric_terms = pace.util.grid.MetricTerms(
            quantity_factory=self.quantity_factory, communicator=self.communicator
        )
        self.grid_data = pace.util.grid.GridData.new_from_metric_terms(
            self.metric_terms
        )

        self.stencil_config = pace.dsl.stencil.StencilConfig(
            compilation_config=pace.dsl.stencil.CompilationConfig(
                backend=backend, rebuild=False, validate_args=False
            ),
        )

        self.grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
            sizer=self.sizer, cube=self.communicator
        )
        self.stencil_factory = pace.dsl.StencilFactory(
            config=self.stencil_config, grid_indexing=self.grid_indexing
        )

        self.dycore_state = fv3core.DycoreState.init_zeros(
            quantity_factory=self.quantity_factory
        )

        damping_coefficients = pace.util.grid.DampingCoefficients.new_from_metric_terms(
            self.metric_terms
        )

        self.dynamical_core = fv3core.DynamicalCore(
            comm=comm,
            grid_data=self.grid_data,
            stencil_factory=self.stencil_factory,
            quantity_factory=self.quantity_factory,
            damping_coefficients=damping_coefficients,
            config=self.config.dycore_config,
            timestep=self.config.timestep,
            phis=self.dycore_state.phis,
            state=self.dycore_state,
        )

    def __call__(self, *args):

        self._fortran_data_to_pace(*args)

        self.dynamical_core.step_dynamics(
            state=self.dycore_state,
        )
        return self._pace_data_to_fortran()

    def _put_fortran_data_in_dycore(self, *args):
        self.dycore_state == self.dycore_state
        pass

    def _get__dycore_state_for_fortran(self):
        state = []
        return state
