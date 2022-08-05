import copy
import dataclasses
import os
from typing import Tuple

import dacite
import f90nml
import xarray as xr
import yaml

import fv3core
import pace.dsl
import pace.util
from fv3core.testing.translate_fvdynamics import TranslateFVDynamics
from pace.stencils.testing import TranslateGrid, dataset_to_dict
from pace.stencils.testing.grid import Grid
from pace.util.grid import DampingCoefficients


def get_grid(data_path: str, rank: int, layout: Tuple[int, int], backend: str) -> Grid:
    ds_grid: xr.Dataset = xr.open_dataset(os.path.join(data_path, "Grid-Info.nc")).isel(
        savepoint=0
    )
    grid = TranslateGrid(
        dataset_to_dict(ds_grid.isel(rank=rank)),
        rank=rank,
        layout=layout,
        backend=backend,
    ).python_grid()
    return grid


def test_fv_dynamics(
    backend: str, data_path: str, calibrate_thresholds: bool, threshold_path: str
):
    namelist = pace.util.Namelist.from_f90nml(
        f90nml.read(os.path.join(data_path, "input.nml"))
    )
    threshold_filename = os.path.join(threshold_path, "fv_dynamics.yaml")
    communicator = pace.util.CubedSphereCommunicator(
        comm=pace.util.MPIComm(),
        partitioner=pace.util.CubedSpherePartitioner(
            tile=pace.util.TilePartitioner(layout=namelist.layout)
        ),
    )
    stencil_factory = pace.dsl.StencilFactory(
        config=pace.dsl.StencilConfig(
            compilation_config=pace.dsl.CompilationConfig(
                backend=backend,
                communicator=communicator,
            )
        ),
        grid_indexing=pace.dsl.GridIndexing.from_sizer_and_communicator(
            sizer=pace.util.SubtileGridSizer(), cube=communicator
        ),
    )
    grid = get_grid(
        data_path=data_path,
        rank=communicator.rank,
        layout=namelist.layout,
        backend=backend,
    )
    translate = TranslateFVDynamics(
        grid=grid, namelist=namelist, stencil_factory=stencil_factory
    )
    ds = xr.open_dataset(os.path.join(data_path, "FVDynamics-In.nc")).sel(
        savepoint=0, rank=communicator.rank
    )
    input_data = dataset_to_dict(ds)
    state, grid_data = translate.prepare_data(input_data)
    dycore_config = fv3core.DynamicalCoreConfig.from_namelist(namelist)
    # TODO: refactor so dycore.update_state is no longer a method
    # and DycoreState is a statically-typed class and we don't have to make
    # a dycore until we're actually running the simulation
    init_dycore = fv3core.DynamicalCore(
        comm=communicator,
        grid_data=grid_data,
        stencil_factory=stencil_factory,
        damping_coefficients=grid.damping_coefficients,
        config=dycore_config,
        phis=state.phis,
        state=state,
    )
    init_dycore.update_state(
        namelist.consv_te,
        input_data["do_adiabatic_init"],
        input_data["bdt"],
        namelist.n_split,
        state,
    )
    if calibrate_thresholds:
        thresholds = _calibrate_thresholds(
            state=state,
            communicator=communicator,
            grid_data=grid_data,
            stencil_factory=stencil_factory,
            damping_coefficients=grid.damping_coefficients,
            dycore_config=dycore_config,
            n_trials=10,
            factor=2.0,
        )
        with open(threshold_filename, "w") as f:
            yaml.safe_dump(dataclasses.asdict(thresholds), f)
    with open(threshold_filename, "r") as f:
        data = yaml.safe_load(f)
        thresholds = dacite.from_dict(
            data_class=pace.util.SavepointThresholds,
            data=data,
            config=dacite.Config(strict=True),
        )
    validation = pace.util.ValidationCheckpointer(
        savepoint_data_path=data_path, thresholds=thresholds, rank=communicator.rank
    )
    dycore = fv3core.DynamicalCore(
        comm=communicator,
        grid_data=grid_data,
        stencil_factory=stencil_factory,
        damping_coefficients=grid.damping_coefficients,
        config=dycore_config,
        phis=state.phis,
        state=state,
        checkpointer=validation,
    )
    with validation.trial():
        dycore.step_dynamics(state)


def _calibrate_thresholds(
    state: fv3core.DycoreState,
    communicator: pace.util.CubedSphereCommunicator,
    grid_data: fv3core.GridData,
    stencil_factory: pace.dsl.StencilFactory,
    damping_coefficients: DampingCoefficients,
    dycore_config: fv3core.DynamicalCoreConfig,
    n_trials: int,
    factor: float,
):
    calibration = pace.util.ThresholdCalibrationCheckpointer(factor=factor)
    dycore = fv3core.DynamicalCore(
        comm=communicator,
        grid_data=grid_data,
        stencil_factory=stencil_factory,
        damping_coefficients=damping_coefficients,
        config=dycore_config,
        phis=state.phis,
        state=state,
    )
    original_state = copy.deepcopy(state)
    for _ in range(n_trials):
        with calibration.trial():
            dycore.step_dynamics(copy.deepcopy(original_state))
    return calibration.thresholds
