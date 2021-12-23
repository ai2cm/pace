import os
import sys
from dataclasses import fields
from typing import List, Literal, Optional, Tuple

import click
import f90nml
import numpy as np
from mpi4py import MPI

import fv3core._config
import fv3core.initialization.baroclinic as baroclinic_init
import pace.util
from fv3core import DynamicalCore
from fv3core.initialization.dycore_state import DycoreState
from fv3core.testing.translate_fvdynamics import init_dycore_state_from_serialized_data
from fv3gfs.physics import PhysicsState
from fv3gfs.physics.stencils.physics import Physics
from pace.dsl.stencil import StencilFactory
from pace.stencils.testing.grid import DampingCoefficients, DriverGridData, GridData
from pace.stencils.update_atmos_state import UpdateAtmosphereState
from pace.util import QuantityFactory
from pace.util.constants import N_HALO_DEFAULT
from pace.util.grid import MetricTerms


class Driver:
    def __init__(
        self,
        namelist: fv3core._config.Namelist,
        comm,
        backend: str,
        physics_packages: List[str],
        dycore_init_mode="baroclinic",
        rebuild: bool = False,
        validate_args: bool = True,
        data_dir: Optional[str] = None,
    ):
        """Initializes a pace Driver
        Args:
          namelist: model configuration information
          comm: communication object behaving like mpi4py.Comm
          backend: gt4py backend name
          physics_packages: a list of names of physics packages to turn on
          dycore_init_mode: dycore state init option serialized_data
                            or baroclinic
          rebuild: specify whether to force recompile all gt4py stencils
                   or reuse previously compiled stencils when they exist
          validate_args: gt4py option on whether to check stencil arguments
          data_dir: optional path to serialized data for reading in initial state
                    from Fortran data.

        """
        self._data_dir = data_dir
        self._namelist = namelist
        self._backend = backend
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(self._namelist.layout)
        )
        self._comm = pace.util.CubedSphereCommunicator(comm, partitioner)
        self._quantity_factory, self._stencil_factory = self._setup_factories(
            rebuild, validate_args
        )
        self._metric_terms, self._grid_data = self._compute_grid()
        self.dycore_state = self._initialize_dycore_state(dycore_init_mode)

        self.dycore = DynamicalCore(
            comm=self._comm,
            grid_data=self._grid_data,
            stencil_factory=self._stencil_factory,
            damping_coefficients=DampingCoefficients.new_from_metric_terms(
                self._metric_terms
            ),
            config=self._namelist.dynamical_core,
            phis=self.dycore_state.phis,
        )
        if not self._namelist.dycore_only:
            self.physics = Physics(
                stencil_factory=self._stencil_factory,
                grid_data=self._grid_data,
                namelist=self._namelist,
                active_packages=physics_packages,
            )
            self.physics_state = self._init_physics_state_from_dycore_state(
                physics_packages
            )
            self.state_updater = UpdateAtmosphereState(
                stencil_factory=self._stencil_factory,
                grid_data=self._grid_data,
                namelist=self._namelist,
                comm=self._comm,
                grid_info=DriverGridData.new_from_metric_terms(self._metric_terms),
                quantity_factory=self._quantity_factory,
            )

    def _setup_factories(
        self, rebuild: bool, validate_args: bool
    ) -> Tuple["QuantityFactory", "StencilFactory"]:
        stencil_config = pace.dsl.stencil.StencilConfig(
            backend=self._backend,
            rebuild=rebuild,
            validate_args=validate_args,
        )
        sizer = pace.util.SubtileGridSizer.from_tile_params(
            nx_tile=self._namelist.npx - 1,
            ny_tile=self._namelist.npy - 1,
            nz=self._namelist.npz,
            n_halo=N_HALO_DEFAULT,
            extra_dim_lengths={},
            layout=self._namelist.layout,
            tile_partitioner=self._comm.partitioner.tile,
            tile_rank=self._comm.tile.rank,
        )

        grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
            sizer=sizer, cube=self._comm
        )
        quantity_factory = QuantityFactory.from_backend(sizer, backend=self._backend)
        stencil_factory = StencilFactory(
            config=stencil_config,
            grid_indexing=grid_indexing,
        )
        return quantity_factory, stencil_factory

    def _compute_grid(self) -> Tuple["MetricTerms", "GridData"]:
        metric_terms = MetricTerms(
            quantity_factory=self._quantity_factory, communicator=self._comm
        )
        grid_data = GridData.new_from_metric_terms(metric_terms)
        return metric_terms, grid_data

    def _initialize_dycore_state(
        self, dycore_init_mode: Literal["serialized_data", "baroclinic"]
    ) -> "DycoreState":
        if dycore_init_mode == "serialized_data":
            sys.path.append("/usr/local/serialbox/python/")
            import serialbox

            serializer = serialbox.Serializer(
                serialbox.OpenModeKind.Read,
                self._data_dir,
                "Generator_rank" + str(self._comm.rank),
            )
            dycore_state = init_dycore_state_from_serialized_data(
                serializer=serializer,
                rank=self._comm.rank,
                backend=self._backend,
                namelist=self._namelist.dynamical_core,
                quantity_factory=self._quantity_factory,
                stencil_factory=self._stencil_factory,
            )

        elif dycore_init_mode == "baroclinic":
            # create an initial state from the Jablonowski & Williamson Baroclinic
            # test case perturbation. JRMS2006

            dycore_state = baroclinic_init.init_baroclinic_state(
                self._metric_terms,
                adiabatic=self._namelist.adiabatic,
                hydrostatic=self._namelist.hydrostatic,
                moist_phys=self._namelist.moist_phys,
                comm=self._comm,
            )
        else:
            raise ValueError(
                "dycore_init_mode is "
                + dycore_init_mode
                + ", which is not one of the supported options"
            )
        return dycore_state

    def _init_physics_state_from_dycore_state(
        self, physics_packages: List[str]
    ) -> "PhysicsState":
        initial_storages = {}
        dycore_fields = fields(DycoreState)
        for field in fields(PhysicsState):
            metadata = field.metadata
            matches = [
                f
                for f in dycore_fields
                if field.name == f.name
                and metadata["name"] == f.metadata["name"]
                and metadata["units"] == f.metadata["units"]
            ]
            if len(matches) > 0:
                initial_storages[field.name] = getattr(self.dycore_state, field.name)
            else:
                initial_storages[field.name] = self._quantity_factory.zeros(
                    [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
                    field.metadata["units"],
                    dtype=float,
                )
        return PhysicsState(
            **initial_storages,
            quantity_factory=self._quantity_factory,
            active_packages=physics_packages,
        )

    def step(self, do_adiabatic_init, timestep):
        self.step_dynamics(do_adiabatic_init, timestep)
        if not self._namelist.dycore_only:
            self.step_physics()

    def step_dynamics(self, do_adiabatic_init, timestep):
        self.dycore.step_dynamics(
            state=self.dycore_state,
            conserve_total_energy=self._namelist.consv_te,
            n_split=self._namelist.n_split,
            do_adiabatic_init=do_adiabatic_init,
            timestep=timestep,
        )

    def step_physics(self):
        self.physics(self.physics_state)
        self.state_updater(self.dycore_state, self.physics_state)


MODEL_OUT_DIR = "./model_output"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


@click.command()
@click.argument(
    "data_directory",
    required=False,
    nargs=1,
    default="/port_dev/test_data/c12_6ranks_baroclinic_dycore_microphysics",
)
@click.argument("time_steps", required=False, default="1")
@click.argument("backend", required=False, default="numpy")
@click.argument("init_mode", required=False, default="baroclinic")
def driver(data_directory: str, time_steps: str, backend: str, init_mode: str):

    experiment_name = os.path.basename(data_directory)
    print("Running configuration " + experiment_name)

    namelist = fv3core._config.Namelist.from_f90nml(
        f90nml.read(data_directory + "/input.nml")
    )
    output_vars = [
        "u",
        "v",
        "ua",
        "va",
        "pt",
        "delp",
        "qvapor",
        "qliquid",
        "qice",
        "qrain",
        "qsnow",
        "qgraupel",
    ]
    physics_packages = ["microphysics"]
    driver = Driver(
        namelist,
        comm,
        backend,
        physics_packages,
        dycore_init_mode=init_mode,
        data_dir=data_directory,
    )
    # TODO: add logic to support when this is True
    do_adiabatic_init = False
    # TODO derive from namelist
    bdt = 225.0

    for t in range(int(time_steps)):
        driver.step(
            do_adiabatic_init=do_adiabatic_init,
            timestep=bdt,
        )

        if t % 5 == 0:
            comm.Barrier()

            output = {}

            for key in output_vars:
                getattr(driver.dycore_state, key).synchronize()
                output[key] = np.asarray(getattr(driver.dycore_state, key))
            np.save("pace_output_t_" + str(t) + "_rank_" + str(rank) + ".npy", output)


if __name__ == "__main__":
    # Make sure the model output directory exists
    from pathlib import Path

    Path(MODEL_OUT_DIR).mkdir(parents=True, exist_ok=True)
    # Run the experiment
    driver()
