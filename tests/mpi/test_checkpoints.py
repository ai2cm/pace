import dataclasses
import os
from typing import List, Tuple

import dacite
import f90nml
import xarray as xr
import yaml

import pace.dsl
import pace.util
from pace import fv3core
from pace.fv3core.initialization.dycore_state import DycoreState
from pace.fv3core.testing.translate_fvdynamics import TranslateFVDynamics
from pace.stencils.testing import TranslateGrid, dataset_to_dict
from pace.stencils.testing.grid import Grid
from pace.util.checkpointer.thresholds import SavepointThresholds
from pace.util.grid import DampingCoefficients, GridData
from pace.util.testing import perturb


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


class StateInitializer:
    def __init__(
        self,
        ds: xr.Dataset,
        translate: TranslateFVDynamics,
        communicator,
        stencil_factory,
        grid,
        dycore_config,
        consv_te,
        n_split,
    ):
        self._ds = ds
        self._translate = translate
        self._consv_te = consv_te
        self._n_split = n_split

        input_data = dataset_to_dict(self._ds.copy())
        state, grid_data = self._translate.prepare_data(input_data)
        self._dycore = fv3core.DynamicalCore(
            comm=communicator,
            grid_data=grid_data,
            stencil_factory=stencil_factory,
            damping_coefficients=grid.damping_coefficients,
            config=dycore_config,
            phis=state.phis,
            state=state,
        )

    def new_state(self) -> Tuple[DycoreState, GridData]:
        input_data = dataset_to_dict(self._ds.copy())
        state, grid_data = self._translate.prepare_data(input_data)

        self._dycore.update_state(
            self._consv_te,
            input_data["do_adiabatic_init"],
            input_data["bdt"],
            self._n_split,
            state,
        )
        return state, grid_data


def test_fv_dynamics(
    backend: str, data_path: str, calibrate_thresholds: bool, threshold_path: str
):
    print("start test call")
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
                rebuild=False,
            )
        ),
        grid_indexing=pace.dsl.GridIndexing.from_sizer_and_communicator(
            sizer=pace.util.SubtileGridSizer.from_tile_params(
                nx_tile=namelist.npx - 1,
                ny_tile=namelist.npy - 1,
                nz=namelist.npz,
                n_halo=3,
                tile_partitioner=communicator.partitioner.tile,
                tile_rank=communicator.rank,
                extra_dim_lengths={},
                layout=namelist.layout,
            ),
            cube=communicator,
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
    dycore_config = fv3core.DynamicalCoreConfig.from_namelist(namelist)
    initializer = StateInitializer(
        ds,
        translate,
        communicator,
        stencil_factory,
        grid,
        dycore_config,
        namelist.consv_te,
        namelist.n_split,
    )
    print("stencils initialized")
    if calibrate_thresholds:
        thresholds = _calibrate_thresholds(
            initializer=initializer,
            communicator=communicator,
            stencil_factory=stencil_factory,
            damping_coefficients=grid.damping_coefficients,
            dycore_config=dycore_config,
            n_trials=10,
            factor=10.0,
        )
        print(f"calibrated thresholds: {thresholds}")
        if communicator.rank == 0:
            with open(threshold_filename, "w") as f:
                yaml.safe_dump(dataclasses.asdict(thresholds), f)
        communicator.comm.barrier()
    print("thresholds calibrated")
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
    state, grid_data = initializer.new_state()
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
    initializer: StateInitializer,
    communicator: pace.util.CubedSphereCommunicator,
    stencil_factory: pace.dsl.StencilFactory,
    damping_coefficients: DampingCoefficients,
    dycore_config: fv3core.DynamicalCoreConfig,
    n_trials: int,
    factor: float,
):
    state, grid_data = initializer.new_state()
    calibration = pace.util.ThresholdCalibrationCheckpointer(factor=factor)
    dycore = fv3core.DynamicalCore(
        comm=communicator,
        grid_data=grid_data,
        stencil_factory=stencil_factory,
        damping_coefficients=damping_coefficients,
        config=dycore_config,
        phis=state.phis,
        state=state,
        checkpointer=calibration,
    )
    for i in range(n_trials):
        print(f"running calibration trial {i}")
        trial_state, _ = initializer.new_state()
        perturb(dycore_state_to_dict(trial_state))
        with calibration.trial():
            dycore.step_dynamics(trial_state)
    all_thresholds = communicator.comm.allgather(calibration.thresholds)
    thresholds = merge_thresholds(all_thresholds)
    set_manual_thresholds(thresholds)
    return thresholds


def set_manual_thresholds(thresholds: SavepointThresholds):
    # all thresholds on the input data are 0 because no computation has happened yet
    for entry in thresholds.savepoints["FVDynamics-In"]:
        for name in entry:
            entry[name] = pace.util.Threshold(relative=0.0, absolute=0.0)


def merge_thresholds(all_thresholds: List[pace.util.SavepointThresholds]):
    thresholds = all_thresholds[0]
    for other_thresholds in all_thresholds[1:]:
        for savepoint_name in thresholds.savepoints:
            for i_call in range(len(thresholds.savepoints[savepoint_name])):
                for variable_name in thresholds.savepoints[savepoint_name][i_call]:
                    thresholds.savepoints[savepoint_name][i_call][
                        variable_name
                    ] = thresholds.savepoints[savepoint_name][i_call][
                        variable_name
                    ].merge(
                        other_thresholds.savepoints[savepoint_name][i_call][
                            variable_name
                        ]
                    )
    return thresholds


def dycore_state_to_dict(state: DycoreState):
    return {
        name: getattr(state, name).data
        for name in dir(state)
        if isinstance(getattr(state, name), pace.util.Quantity)
    }
