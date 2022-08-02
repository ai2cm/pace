import copy as cp
import enum
import os
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube
from IPython.display import HTML, display
from matplotlib import animation
from netCDF4 import Dataset

from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from fv3core.stencils.tracer_2d_1l import TracerAdvection
from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.stencil import GridIndexing, StencilConfig, StencilFactory
from pace.util import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    Quantity,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from pace.util.constants import RADIUS
from pace.util.grid import DampingCoefficients, GridData, MetricTerms
from pace.util.grid.gnomonic import great_circle_distance_lon_lat


class GridType(enum.Enum):
    AGrid = 1
    CGrid = 2
    DGrid = 3


backend = "numpy"
density = 1
layout = (1, 1)
nhalo = 3
pressure_base = 10
tracer_base = 1.0
fvt_dict = {"grid_type": 0, "hord": 6}


def store_namelist_variables(local_variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use: namelistDict =
            store_namelist_variables(local_variables)

    Stores namelist variables into a dictionary.

    Inputs:
    - locally defined variables (locals())

    Outputs:
    - namelistDict (Dict)
    """

    namelist_variables = [
        "nx",
        "ny",
        "timestep",
        "nDays",
        "test_case",
        "plot_outputDuring",
        "plot_outputAfter",
        "plot_jupyterAnimation",
        "figure_everyNhours",
        "write_initialCondition",
        "plot_gridLayout",
        "show_figures",
        "tracer_center",
        "nSeconds",
        "figure_everyNsteps",
        "nSteps_advection",
    ]

    namelistDict = {}
    for i in namelist_variables:
        namelistDict[i] = local_variables[i]

    return namelistDict


def define_metadata(
    namelistDict: Dict[str, Any], mpi_comm: Any
) -> Tuple[Dict[str, int], Dict[str, Tuple[Any]], Dict[str, str], int]:
    """
    Use: metadata, mpi_rank =
            define_metadata(namelistDict, mpi_comm)

    Creates dictionaries with metadata used for Quantity definitions.

    Inputs:
    - namelistDict (Dict) from store_namelist_variables()
    - mpi_comm

    Outputs:
    - metadata (Dict):
        - dimensions (Dict)
        - origins (Dict)
        - units (Dict)
    - mpi_rank
    """

    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    # to match fortran
    nx = namelistDict["nx"] + 1
    ny = namelistDict["ny"] + 1
    nz = 79 + 1

    dimensions = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "nx1": nx + 1,
        "ny1": ny + 1,
        "nhalo": nhalo,
        "nxhalo": nx + 2 * nhalo,
        "nyhalo": ny + 2 * nhalo,
        "tile": mpi_size,
    }

    units = {
        "areaflux": "m2",
        "coord": "degrees",
        "courant": "",
        "dist": "m",
        "mass": "kg",
        "pressure": "Pa",
        "psi": "kg/m/s",
        "wind": "m/s",
    }

    origins = {
        "halo": (0, 0),
        "compute_2d": (dimensions["nhalo"], dimensions["nhalo"]),
        "compute_3d": (dimensions["nhalo"], dimensions["nhalo"], 0),
    }

    metadata = {"dimensions": dimensions, "origins": origins, "units": units}

    return metadata, mpi_rank


def split_metadata(
    metadata: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, int], Dict[str, Tuple[Any]], Dict[str, str]]:
    """
    Use: dimensions, origins, units =
            split.metadata(metadata)

    Splits the metadata dictionary into its component dictionaries.

    Inputs:
    - metadata (Dict) from define_metadata()

    Outputs:
    - dimensions (Dict)
    - origins (Dict)
    - units (Dict)
    """

    dimensions = metadata["dimensions"]
    origins = metadata["origins"]
    units = metadata["units"]

    return dimensions, origins, units


def configure_domain(mpi_comm: Any, dimensions: Dict[str, int]) -> Dict[str, Any]:
    """
    Use: configuration =
            configure_domain(mpi_comm, dimensions)

    Creates all domain configuration parameters and stores them.

    Inputs:
    - mpi_comm: communicator
    - dimensions (Dict)

    Outputs:
    - configuration (Dict):
        - partitioner
        - communicator
        - sizer
        - quantity_factory
        - metric_terms
        - damping_coefficients
        - grid_data
        - dace_config
        - stencil_config
        - grid_indexing
        - stencil_factory
    """

    partitioner = CubedSpherePartitioner(TilePartitioner(layout))
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=dimensions["nx"] - 1,
        ny_tile=dimensions["ny"] - 1,
        nz=dimensions["nz"] - 1,
        n_halo=dimensions["nhalo"],
        extra_dim_lengths={},
        layout=layout,
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)

    metric_terms = MetricTerms(
        quantity_factory=quantity_factory, communicator=communicator
    )
    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
    grid_data = GridData.new_from_metric_terms(metric_terms)

    dace_config = DaceConfig(
        communicator=None, backend=backend, orchestration=DaCeOrchestration.Python
    )
    stencil_config = StencilConfig(
        backend=backend, rebuild=False, validate_args=True, dace_config=dace_config
    )
    grid_indexing = GridIndexing.from_sizer_and_communicator(
        sizer=sizer, cube=communicator
    )

    # set the domain so there is only one level in the vertical -- forced
    domain = grid_indexing.domain
    domain_new = list(domain)
    domain_new[2] = 1
    domain_new = tuple(domain_new)

    grid_indexing.domain = domain_new

    stencil_factory = StencilFactory(config=stencil_config, grid_indexing=grid_indexing)

    configuration = {
        "partitioner": partitioner,
        "communicator": communicator,
        "sizer": sizer,
        "quantity_factory": quantity_factory,
        "metric_terms": metric_terms,
        "damping_coefficients": damping_coefficients,
        "grid_data": grid_data,
        "dace_config": dace_config,
        "stencil_config": stencil_config,
        "grid_indexing": grid_indexing,
        "stencil_factory": stencil_factory,
    }

    return configuration


def get_lon_lat_edges(
    configuration: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    gather: bool = True,
) -> Tuple[Quantity, Quantity]:
    """
    Use: lon, lat =
    get_lon_lat_edges(configuration, metadata, gather=True)

    Creates quantities containing longitude and latitude of
    tile edges (without halo points), in degrees.

    Inputs:
    - configuration (Dict) from configure_domain()
    - metadata (Dict)
    - gather (bool): if true, then gathers all tiles

    Outputs:
    - lon in degrees
    - lat in degrees
    """

    dimensions, origins, units = split_metadata(metadata)

    lon = Quantity(
        configuration["grid_data"].lon.data * 180 / np.pi,
        ("x_interface", "y_interface"),
        units["coord"],
        origins["compute_2d"],
        (dimensions["nx1"], dimensions["ny1"]),
        backend,
    )
    lat = Quantity(
        configuration["grid_data"].lat.data * 180 / np.pi,
        ("x_interface", "y_interface"),
        units["coord"],
        origins["compute_2d"],
        (dimensions["nx1"], dimensions["ny1"]),
        backend,
    )

    if gather:
        lon = configuration["communicator"].gather(lon)
        lat = configuration["communicator"].gather(lat)

    return lon, lat


def create_gaussian_multiplier(
    lon: np.ndarray,
    lat: np.ndarray,
    dimensions: Dict[str, int],
    center: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Use: gaussian_multiplier =
            create_gaussian_multiplier(lon, lat, dimensions, mpi_rank, target_tile)

    Calculates a gaussian-bell shaped multiplier for tracer initialization.
    It centers the bell in the middle of the target_tile provided.

    Inputs:
    - lon and lat (in radians, and including halo points)
    - dimensions (Dict)
    - mpi_rank
    - target_tile

    Outputs:
    - gaussian_multiplier: array of values between 0 and 1
    """

    r0 = RADIUS / 3.0
    gaussian_multiplier = np.zeros((dimensions["nxhalo"], dimensions["nyhalo"]))
    p_center = [np.deg2rad(center[0]), np.deg2rad(center[1])]

    for jj in range(dimensions["nyhalo"]):
        for ii in range(dimensions["nxhalo"]):

            p_dist = [lon[ii, jj], lat[ii, jj]]
            r = great_circle_distance_lon_lat(
                p_center[0], p_dist[0], p_center[1], p_dist[1], RADIUS, np
            )

            gaussian_multiplier[ii, jj] = (
                0.5 * (1.0 + np.cos(np.pi * r / r0)) if r < r0 else 0.0
            )

    return gaussian_multiplier


def calculate_streamfunction(
    lonA: np.ndarray,
    latA: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    dimensions: Dict[str, int],
    test_case: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use: psi, psi_staggered =
            calculate_streamfunction(lonA, latA, lon, lat, dimensions, test_case)

    Creates streamfunction from input quantities, for defined test cases:
        - a) constant radius (as in Fortran test case 1)
        - b) radius varies with latitude, less spreading out.
        - c) south-north propagation from tile 1

    Inputs:
    - lonA, latA (in radians) on A-grid
    - lon, lat (in radians) on cell corners
    - dimensions
    - test_case ("a" or "b")

    Outputs:
    - psi (streamfunction)
    - psi_staggered (streamfunction, but on tile corners.)
    """

    xA_t = np.cos(latA) * np.cos(lonA)
    yA_t = np.cos(latA) * np.sin(lonA)
    zA_t = np.sin(latA)
    x_t = np.cos(lat) * np.cos(lon)
    y_t = np.cos(lat) * np.sin(lon)
    z_t = np.sin(lat)

    if test_case == "a":
        RadA = RADIUS * np.ones(lonA.shape)
        Rad = RADIUS * np.ones(lon.shape)
        multiplierA = zA_t
        multiplier = z_t
    elif test_case == "b":
        RadA = RADIUS * np.cos(latA / 2)
        Rad = RADIUS * np.cos(lat / 2)
        multiplierA = zA_t
        multiplier = z_t
    elif test_case == "c":
        RadA = RADIUS
        Rad = RADIUS
        multiplierA = -yA_t
        multiplier = -y_t
    elif test_case == "d":
        print("This case is in TESTING.")
        RadA = RADIUS * np.ones(lon.shape)
        Rad = RADIUS * np.ones(lon.shape)
        multiplierA = (yA_t + zA_t) / 1.5
        multiplier = (y_t + z_t) / 1.5
    else:
        RadA = np.ones(lonA.shape) * np.nan
        Rad = np.ones(lon.shape) * np.nan
        multiplierA = np.nan
        multiplier = np.nan
        print("Please choose one of the defined test cases.")
        print("This will return gibberish.")

    Ubar = (2.0 * np.pi * RADIUS) / (12.0 * 86400.0)
    psi = -1 * Ubar * RadA * multiplierA
    psi_staggered = -1 * Ubar * Rad * multiplier

    # original multiplier = np.sin(latA[ii, jj]) * np.cos(alpha)
    #                 - np.cos(lonA[ii, jj]) * np.cos(latA[ii, jj]) * np.sin(alpha)
    # but since alpha = 0, we don't need to worry about anything.

    return psi, psi_staggered


def calculate_winds_from_streamfunction_grid(
    psi: np.ndarray,
    dx: Quantity,
    dy: Quantity,
    dimensions: Dict[str, int],
    grid: GridType = GridType.AGrid,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use: u_grid, v_grid =
            calculate_winds_from_streamfunction_grid(psi, dx, dy, dimensions, grid)


    Returns winds on a chosen grid based on streamfunction and grid spacing.

    Inputs:
    - psi: streamfunction
    - dx, dy: distance between points
    - dimensions(Dict))

    Outputs:
    - u_grid: x-direction wind on chosen grid
    - v_grid: y-direction wind on chosen grid

    Grid options:
    - A: A-grid, center points
    - C: C-grid, edge points, (y dim + 1 for u, x dim +1 for v)
    - D: D-grid, edge points (x dim + 1 for u, y dim +1 for v)

    For different grid, input functions are different:
    - A: streamfunction, dx, dy on cell centers, all with halos
    - C: streamfunction on corner points, dx, dy on edge points, all with halos
    - D: streamfunction on center points, dx, dy on edge points, all with halos
    """
    u_grid = np.zeros((dimensions["nxhalo"], dimensions["nyhalo"]))
    v_grid = np.zeros((dimensions["nxhalo"], dimensions["nyhalo"]))

    if grid == GridType.AGrid:
        for jj in range(
            dimensions["nhalo"] - 1, dimensions["ny"] + dimensions["nhalo"] + 1
        ):
            for ii in range(
                dimensions["nhalo"] - 1, dimensions["nx"] + dimensions["nhalo"] + 1
            ):
                psi1 = 0.5 * (psi.data[ii, jj] + psi.data[ii, jj - 1])
                psi2 = 0.5 * (psi.data[ii, jj] + psi.data[ii, jj + 1])
                dist = dy.data[ii, jj]
                u_grid[ii, jj] = 0 if dist == 0 else -1.0 * (psi2 - psi1) / dist

                psi1 = 0.5 * (psi.data[ii, jj] + psi.data[ii - 1, jj])
                psi2 = 0.5 * (psi.data[ii, jj] + psi.data[ii + 1, jj])
                dist = dx.data[ii, jj]
                v_grid[ii, jj] = 0 if dist == 0 else (psi2 - psi1) / dist

    elif grid == GridType.CGrid:
        v_grid = np.zeros(psi.data.shape) * np.nan
        v_grid[:-1] = (psi.data[1:] - psi.data[:-1]) / dx.data[:-1]

        u_grid = np.zeros(psi.data.shape) * np.nan
        u_grid[:, :-1] = -1 * (psi.data[:, 1:] - psi.data[:, :-1]) / dy.data[:, :-1]

    elif grid == GridType.DGrid:
        for jj in range(
            dimensions["nhalo"] - 1, dimensions["ny"] + dimensions["nhalo"] + 1
        ):
            for ii in range(
                dimensions["nhalo"] - 1, dimensions["nx"] + dimensions["nhalo"] + 1
            ):
                dist = dx.data[ii, jj]
                v_grid[ii, jj] = (
                    0 if dist == 0 else (psi.data[ii, jj] - psi.data[ii - 1, jj]) / dist
                )

                dist = dy.data[ii, jj]
                u_grid[ii, jj] = (
                    0
                    if dist == 0
                    else -1.0 * (psi.data[ii, jj] - psi.data[ii, jj - 1]) / dist
                )

    return u_grid, v_grid


def create_initial_state(
    grid_data: GridData,
    metadata: Dict[str, Any],
    tracer_center: Tuple[float, float],
    test_case: str,
) -> Dict[str, Quantity]:
    """
    Use: initialState =
            create_initial_state(grid_data, metadata, tracer_center,
            test_case)

    Creates the initial state for one of the defined test cases:
        - a) constant radius (as in Fortran test case 1)
        - b) radius varies with latitude, less spreading out.

    Goes through the following steps:
        - creates lon and lat on all points (including halos)
        - creates tracer initial state by calling gaussian_multiplier()
        - creates pressure initial state by imposing constant pressure_base
        - calculates streamfunction by calling calculate_streamfunction()
        - get intial winds on c-grid by calling
            calculate_winds_from_streamfunction_grid()
        - extends initial fields into a single vertical layer

    Inputs:
    - grid_data (cofiguration) from configure_domain()
    - metadata (Dict) from define_metadata()
    - tracerCenter
    - test_case ("a" or "b")

    Outputs:
    - initialState (Dict):
        - delp
        - tracer
        - uC
        - vC
    """

    dimensions, origins, units = split_metadata(metadata)

    lonA_halo = Quantity(
        grid_data.lon_agrid.data,
        ("x_halo", "y_halo"),
        units["coord"],
        origins["halo"],
        (dimensions["nxhalo"], dimensions["nyhalo"]),
        backend,
    )
    latA_halo = Quantity(
        grid_data.lat_agrid.data,
        ("x_halo", "y_halo"),
        units["coord"],
        origins["halo"],
        (dimensions["nxhalo"], dimensions["nyhalo"]),
        backend,
    )

    lon_halo = Quantity(
        grid_data.lon.data,
        ("x_halo", "y_halo"),
        units["coord"],
        origins["halo"],
        (dimensions["nxhalo"], dimensions["nyhalo"]),
        backend,
    )
    lat_halo = Quantity(
        grid_data.lat.data,
        ("x_halo", "y_halo"),
        units["coord"],
        origins["halo"],
        (dimensions["nxhalo"], dimensions["nyhalo"]),
        backend,
    )

    # tracer
    gaussian_multiplier = create_gaussian_multiplier(
        lonA_halo.data,
        latA_halo.data,
        dimensions,
        tracer_center,
    )
    tracer = Quantity(
        gaussian_multiplier * tracer_base,
        ("x", "y"),
        units["mass"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny"]),
        backend,
    )

    # pressure
    delp = Quantity(
        np.ones(tracer.data.shape) * pressure_base,
        ("x", "y"),
        units["pressure"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny"]),
        backend,
    )

    # streamfunction
    _, psi_staggered = calculate_streamfunction(
        lonA_halo.data,
        latA_halo.data,
        lon_halo.data,
        lat_halo.data,
        dimensions,
        test_case,
    )
    psi_staggered_halo = Quantity(
        psi_staggered,
        ("x_halo", "y_halo"),
        units["psi"],
        origins["halo"],
        (dimensions["nxhalo"], dimensions["nyhalo"]),
        backend,
    )

    # winds
    dx_halo = Quantity(
        grid_data.dx.data,
        ("x_halo", "y_halo"),
        units["dist"],
        origins["halo"],
        (dimensions["nxhalo"], dimensions["nyhalo"]),
        backend,
    )
    dy_halo = Quantity(
        grid_data.dy.data,
        ("x_halo", "y_halo"),
        units["dist"],
        origins["halo"],
        (dimensions["nxhalo"], dimensions["nyhalo"]),
        backend,
    )
    grid = GridType.CGrid
    uC, vC = calculate_winds_from_streamfunction_grid(
        psi_staggered_halo, dx_halo, dy_halo, dimensions, grid=grid
    )
    uC = Quantity(
        uC,
        ("x", "y_interface"),
        units["wind"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny1"]),
        backend,
    )
    vC = Quantity(
        vC,
        ("x_interface", "y"),
        units["wind"],
        origins["compute_2d"],
        (dimensions["nx1"], dimensions["ny"]),
        backend,
    )

    # extend initial conditions into one vertical layer
    dimensions["nz"] = 1
    empty = np.zeros((dimensions["nxhalo"], dimensions["nyhalo"], dimensions["nz"] + 1))

    tracer_3d = np.copy(empty)
    uC_3d = np.copy(empty)
    vC_3d = np.copy(empty)
    delp_3d = np.copy(empty)

    tracer_3d[:, :, 0] = tracer.data
    uC_3d[:, :, 0] = uC.data
    vC_3d[:, :, 0] = vC.data
    delp_3d[:, :, 0] = delp.data

    tracer = Quantity(
        tracer_3d,
        ("x", "y", "z"),
        units["mass"],
        origins["compute_3d"],
        (dimensions["nx"], dimensions["ny"], dimensions["nz"]),
        backend,
    )
    uC = Quantity(
        uC_3d,
        ("x", "y_interface", "z"),
        units["wind"],
        origins["compute_3d"],
        (dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        backend,
    )
    vC = Quantity(
        vC_3d,
        ("x_interface", "y", "z"),
        units["wind"],
        origins["compute_3d"],
        (dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        backend,
    )
    delp = Quantity(
        delp_3d,
        ("x", "y", "z"),
        units["wind"],
        origins["compute_3d"],
        (dimensions["nx"], dimensions["ny"], dimensions["nz"]),
        backend,
    )

    initialState = {
        "delp": delp,
        "tracer": tracer,
        "uC": uC,
        "vC": vC,
    }

    return initialState


def run_finite_volume_fluxprep(
    configuration: Dict[str, Any],
    initialState: Dict[str, Quantity],
    metadata: Dict[str, Any],
    timestep: float,
) -> Dict[str, Quantity]:
    """
    Use: fluxPrep =
            run_finite_volume_fluxprep(configuration, initialState,
            metadata, timestep)

    Initializes and runs the FiniteVolumeFluxPrep class to get
    initial states for mass flux, contravariant winds, area flux.

    Inputs:
    - configuration (Dict) from configure_domain()
    - initialState (Dict) from create_initial_state()
    - metadata (Dict) from define_metadata()
    - timestep (float) for advection

    Outputs:
    - fluxPrep (Dict):
        - crx and cry
        - mfxd and mfyd
        - ucv, vcv
        - xaf, yaf
    """

    dimensions, origins, units = split_metadata(metadata)

    # create empty quantities to be filled
    empty = np.zeros((dimensions["nxhalo"], dimensions["nyhalo"], dimensions["nz"] + 1))

    crx = Quantity(
        empty,
        ("x_interface", "y", "z"),
        units["courant"],
        origins["compute_3d"],
        (dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        backend,
    )
    cry = Quantity(
        empty,
        ("x", "y_interface", "z"),
        units["courant"],
        origins["compute_3d"],
        (dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        backend,
    )
    xaf = Quantity(
        empty,
        ("x_interface", "y", "z"),
        units["areaflux"],
        origins["compute_3d"],
        (dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        backend,
    )
    yaf = Quantity(
        empty,
        ("x", "y_interface", "z"),
        units["areaflux"],
        origins["compute_3d"],
        (dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        backend,
    )
    ucv = Quantity(
        empty,
        ("x_interface", "y", "z"),
        units["wind"],
        origins["compute_3d"],
        (dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        backend,
    )
    vcv = Quantity(
        empty,
        ("x", "y_interface", "z"),
        units["wind"],
        origins["compute_3d"],
        (dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        backend,
    )

    # intialize and run
    fvf_prep = FiniteVolumeFluxPrep(
        configuration["stencil_factory"], configuration["grid_data"]
    )

    fvf_prep(
        initialState["uC"], initialState["vC"], crx, cry, xaf, yaf, ucv, vcv, timestep
    )  # THIS WILL MODIFY CREATED QUANTITIES, but not change uc, vc

    mfxd = Quantity(
        empty,
        ("x_interface", "y", "z"),
        units["mass"],
        origins["compute_3d"],
        (dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        backend,
    )
    mfyd = Quantity(
        empty,
        ("x", "y_interface", "z"),
        units["mass"],
        origins["compute_3d"],
        (dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        backend,
    )

    mfxd.data[:] = xaf.data[:] * initialState["delp"].data[:] * density
    mfyd.data[:] = yaf.data[:] * initialState["delp"].data[:] * density

    fluxPrep = {
        "crx": crx,
        "cry": cry,
        "mfxd": mfxd,
        "mfyd": mfyd,
        "ucv": ucv,
        "vcv": vcv,
        "xaf": xaf,
        "yaf": yaf,
    }

    return fluxPrep


def build_tracer_advection(
    configuration: Dict[str, Any], tracers: Dict[str, Quantity]
) -> TracerAdvection:
    """
    Use: tracAdv =
            build_tracer_advection(configuration, tracers)


    Initializes FiniteVolumeTransport and TracerAdvection classes.

    Inputs:
    - configuration (Dict) from configure_domain()
    - tracers (Dict) from initialState created by create_initial_state()

    Outputs:
    - tracAdv - an instance of TracerAdvection class
    """

    fvtp_2d = FiniteVolumeTransport(
        configuration["stencil_factory"],
        configuration["grid_data"],
        configuration["damping_coefficients"],
        fvt_dict["grid_type"],
        fvt_dict["hord"],
    )

    tracAdv = TracerAdvection(
        configuration["stencil_factory"],
        fvtp_2d,
        configuration["grid_data"],
        configuration["communicator"],
        tracers,
    )

    return tracAdv


def prepare_everything_for_advection(
    configuration: Dict[str, Any],
    initialState: Dict[str, Quantity],
    metadata: Dict[str, Any],
    timestep: float,
) -> Tuple[Dict[str, Any], TracerAdvection]:
    """
    Use: tracAdv_data, tracAdv =
        prepare_everything_for_advection(configuration, initialState,
        metadata, timestep)

    Calls run_finite_volume_fluxprep() and build_tracer_advection().

    Inputs:
    - configuration from configure_domain()
    - initialState from create_initial_state()
    - metadata from define_metadata()
    - timestep (float) advection timestep

    Outputs:
    - tracAdv_data (Dict):
        - tracers (Dict)
        - delp
        - mfxd and mfyd
        - crx and cry
    - tracAdv: instance of TracerAdvection class

    """

    tracers = {"tracer": initialState["tracer"]}

    fluxPrep = run_finite_volume_fluxprep(
        configuration,
        initialState,
        metadata,
        timestep,
    )

    tracAdv = build_tracer_advection(configuration, tracers)

    tracAdv_data = {
        "tracers": tracers,
        "delp": initialState["delp"],
        "mfxd": fluxPrep["mfxd"],
        "mfyd": fluxPrep["mfyd"],
        "crx": fluxPrep["crx"],
        "cry": fluxPrep["cry"],
    }

    return tracAdv_data, tracAdv


def run_advection_step_with_reset(
    tracAdv_dataInit: Dict[str, Quantity],
    tracAdv_data: Dict[str, Quantity],
    tracAdv: TracerAdvection,
    timestep,
    mpi_rank: Any = None,
) -> Dict[str, Quantity]:
    """
    Use: tracAdv_data =
            run_advection_step_with_reset(tracAdv_dataInit, tracAdv_data,
            tracAdv, timestep, mpi_rank=None)


    Runs one timestep of tracer advection, which overwrites all
    but delp. Then resets mfxd, mfyd, crx, cry to their initial
    values. This is needed because tracAdv overwrites their
    values with 1/3 of the original, which leads to exponential
    decay of advection if left unchanged.

    Inputs:
    - tracAdv_dataInit (Dict) - initial state data, to which
        all but tracer gets reset to after every step
    - tracAdv_data(Dict) - data that gets updated during tracer advection
    - tracAdv (TracerAdvection) instance of class
    - timestep (float)
    - mpi_rank (int or None): if provided, prints out diagnostics

    Outputs:
    - tracAdv_data (Dict) with updated tracer only
    """

    tmp = cp.deepcopy(tracAdv_data["tracers"])  # pre-advection tracer state

    tracAdv(
        tracAdv_data["tracers"],
        tracAdv_data["delp"],
        tracAdv_data["mfxd"],
        tracAdv_data["mfyd"],
        tracAdv_data["crx"],
        tracAdv_data["cry"],
        timestep,
    )

    tracAdv_data["delp"] = cp.deepcopy(tracAdv_dataInit["delp"])
    tracAdv_data["mfxd"] = cp.deepcopy(tracAdv_dataInit["mfxd"])
    tracAdv_data["mfyd"] = cp.deepcopy(tracAdv_dataInit["mfyd"])
    tracAdv_data["crx"] = cp.deepcopy(tracAdv_dataInit["crx"])
    tracAdv_data["cry"] = cp.deepcopy(tracAdv_dataInit["cry"])

    if mpi_rank == 0:
        diff_timestep = tracAdv_data["tracers"]["tracer"].data - tmp["tracer"].data
        diff_fromInit = (
            tracAdv_data["tracers"]["tracer"].data
            - tracAdv_dataInit["tracers"]["tracer"].data
        )
        print(
            "timestep diff min=%.2e; max=%.2e; from init diff min=%.2e; max=%.2e"
            % (
                np.nanmin(diff_timestep),
                np.nanmax(diff_timestep),
                np.nanmin(diff_fromInit),
                np.nanmax(diff_fromInit),
            )
        )

    return tracAdv_data


def plot_grid(
    configuration: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    mpi_rank: Any,
    fOut: str = "grid_map.png",
    show: bool = False,
) -> None:
    """
    Use: plot_grid(configuration, metadata,
            mpi_rank, fOut="grid_map.png", show=False)

    Creates a Robinson projection and plots grid edges.
    Note -- this is basically useless for more than
    50 points as the cells are too small.

    Inputs:
    - configuration (Dict) from configure_domain()
    - metadata (Dict) from define_metadata()
    - mpi_rank
    - fOut (str): file name to save to
    - show (bool): whether to show image in notebook

    Outputs: saved figure
    """

    lon, lat = get_lon_lat_edges(configuration, metadata, gather=True)

    if mpi_rank == 0:

        field = np.zeros(
            (
                metadata["dimensions"]["tile"],
                metadata["dimensions"]["nx"],
                metadata["dimensions"]["ny"],
            )
        )

        fig = plt.figure(figsize=(8, 4))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection=ccrs.Robinson())
        ax.set_facecolor(".4")

        pcolormesh_cube(
            lat.data,
            lon.data,
            field,
            cmap="bwr",
            vmin=-1,
            vmax=1,
            edgecolor="k",
            linewidth=0.1,
        )

        ax.set_title(
            "Grid map: %s x %s"
            % (metadata["dimensions"]["nx"] - 1, metadata["dimensions"]["ny"] - 1)
        )

        plt.savefig(fOut, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close("all")


def plot_projection_field(
    configuration: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    field: Quantity,
    plotDict: Dict[str, Any],
    mpi_rank: Any,
    fOut: str,
    show: bool = False,
) -> None:
    """
    Use: plot_projection_field(configuration, metadata,
            field, pltoDict, mpi_rank, fOut,
            show=False)

    Creates a Robinson projection and plots provided field.

    Inputs:
    - configuration (Dict) from configure_domain()
    - metadata (Dict) from define_metadata()
    - field (Quantity) - single layer
    - mpi_rank
    - fOut (str): file name to save to
    - show (bool): whether to show image in notebook

    Outputs: saved figure
    """

    lon, lat = get_lon_lat_edges(configuration, metadata, gather=True)

    if mpi_rank == 0:
        if not os.path.isdir(os.path.dirname(fOut)):
            os.mkdir(os.path.dirname(fOut))
        else:
            if os.path.isfile(fOut):
                os.remove(fOut)

        field_plot = np.squeeze(field.data)

        fig = plt.figure(figsize=(8, 4))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection=ccrs.Robinson())
        ax.set_facecolor(".4")

        f1 = pcolormesh_cube(
            lat.data,
            lon.data,
            field_plot,
            cmap=plotDict["cmap"],
            vmin=plotDict["vmin"],
            vmax=plotDict["vmax"],
        )
        plt.colorbar(f1, label=plotDict["units"])

        ax.set_title(plotDict["title"])

        plt.savefig(fOut, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close("all")


def plot_tracer_animation(
    configuration: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    tracer_archive: List[Quantity],
    mpi_rank: int,
    plotDict_tracer: Dict[str, Any],
    figure_everyNsteps: int,
    timestep: float,
    frames: Union[str, int] = "all",
) -> None:
    """
    Use: plot_tracer_animation(configuration, metadata,
            tracer_archive, mpi_rank, plotDict_tracer,
            figure_everyNsteps, timestep, frames="all")

    Plots an interactive animation inside Jupyter notebook.

    Inputs:
    - configuration (Dict) from configure_domain()
    - metadata (Dict) from define_metadata()
    - tracer_archive (List) of tracer states
    - mpi_rank
    - plotDict_tracer (Dict) of plotting settings
    - figure_everyNsteps (int) from initial setup
    - timestep (float) for advection
    - frames ("all" or int) - how many frames to plot

    Outputs: animation
    """

    tracer_global = []
    for step in range(len(tracer_archive)):
        tracer_global.append(configuration["communicator"].gather(tracer_archive[step]))

    lon, lat = get_lon_lat_edges(configuration, metadata, gather=True)

    if mpi_rank == 0:
        tracer_stack = np.squeeze(np.stack(tracer_global))[::figure_everyNsteps]
        if frames == "all":
            frames = int(len(tracer_stack))

        fig = plt.figure(figsize=(8, 4))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection=ccrs.Robinson())
        ax.set_facecolor(".4")

        def animate(step: int):
            ax.clear()
            plotDict_tracer["title"] = "Tracer state @ hour: %.2f" % (
                (step * figure_everyNsteps * timestep) / 60 / 60
            )

            pcolormesh_cube(
                lat.data,
                lon.data,
                tracer_stack[step],
                vmin=plotDict_tracer["vmin"],
                vmax=plotDict_tracer["vmax"],
                cmap=plotDict_tracer["cmap"],
            )
            ax.set_title(plotDict_tracer["title"])

        anim = animation.FuncAnimation(
            fig, animate, frames=frames, interval=400, blit=False
        )

        display(HTML(anim.to_jshtml()))
        plt.close()


def write_initial_condition_tofile(
    fOut: str,
    initialState: Dict[str, Quantity],
    metadata: Dict[str, Dict[str, Any]],
    configuration: Dict[str, Any],
    mpi_rank: int,
) -> None:
    """
    Use: write_initial_condition_tofile(
        fOut, initialState, metadata, configuration, mpi_rank)

    Creates netCDF file with initial conditions.

    Inputs:
    - fOut (str) output netcdf file
    - initialState (Dict) from create_initial_state()
    - metadata (Dict) from define_metadata()
    - configuration (Dict) from configure_domain()
    - mpi_rank

    Outputs: written netCDF file
    """

    dimensions, _, units = split_metadata(metadata)

    lon, lat = get_lon_lat_edges(configuration, metadata, gather=True)

    uC = np.squeeze(configuration["communicator"].gather(initialState["uC"]))
    vC = np.squeeze(configuration["communicator"].gather(initialState["vC"]))
    tracer = np.squeeze(configuration["communicator"].gather(initialState["tracer"]))

    if mpi_rank == 0:
        uC = unstagger_coordinate(uC)
        vC = unstagger_coordinate(vC)

        if not os.path.isdir(os.path.dirname(fOut)):
            os.mkdir(os.path.dirname(fOut))
        else:
            if os.path.isfile(fOut):
                os.remove(fOut)

        data = Dataset(fOut, "w")

        for dim in ["tile", "nx", "ny", "nx1", "ny1"]:
            data.createDimension(dim, dimensions[dim])

        v0 = data.createVariable("lon", "f8", ("tile", "nx1", "ny1"))
        v0[:] = lon.data
        v0.units = units["coord"]
        v0 = data.createVariable("lat", "f8", ("tile", "nx1", "ny1"))
        v0[:] = lat.data
        v0.units = units["coord"]

        v1 = data.createVariable("uC", "f8", ("tile", "nx", "ny"))
        v1[:] = uC.data
        v1.units = units["wind"]
        v1.description = "C-grid u wind, unstaggered"
        v1 = data.createVariable("vC", "f8", ("tile", "nx", "ny"))
        v1[:] = vC.data
        v1.units = units["wind"]
        v1.description = "C-grid v wind, unstaggered"

        v1 = data.createVariable("tracer", "f8", ("tile", "nx", "ny"))
        v1[:] = tracer.data
        v1.units = units["mass"]

        data.close()


def write_coordinate_variables_tofile(
    fOut: str,
    metadata: Dict[str, Dict[str, Any]],
    configuration: Dict[str, Any],
    mpi_rank: int,
) -> None:
    """
    Use: write_coordinate_variables_tofile(
        fOut, metadata, configuration, mpi_rank)

    Creates netCDF file with coordinate variables (dx, dy, etc.).
    To be used for making streamfunction experiments easier.

    Inputs:
    - fOut (str) output netcdf file
    - metadata (Dict) from define_metadata()
    - configuration (Dict) from configure_domain()
    - mpi_rank

    Outputs: written netCDF file
    """

    dimensions, origins, units = split_metadata(metadata)

    lon, lat = get_lon_lat_edges(configuration, metadata, gather=True)

    lonA = Quantity(
        configuration["grid_data"].lon_agrid.data * 180 / np.pi,
        ("x", "y"),
        units["coord"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny"]),
        backend,
    )
    latA = Quantity(
        configuration["grid_data"].lat_agrid.data * 180 / np.pi,
        ("x", "y"),
        units["coord"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny"]),
        backend,
    )

    dx = Quantity(
        configuration["grid_data"].dx.data,
        ("x", "y_interface"),
        units["dist"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny1"]),
        backend,
    )
    dy = Quantity(
        configuration["grid_data"].dy.data,
        ("x_interface", "y"),
        units["dist"],
        origins["compute_2d"],
        (dimensions["nx1"], dimensions["ny"]),
        backend,
    )

    lonA = np.squeeze(configuration["communicator"].gather(lonA))
    latA = np.squeeze(configuration["communicator"].gather(latA))
    dx = np.squeeze(configuration["communicator"].gather(dx))
    dy = np.squeeze(configuration["communicator"].gather(dy))

    if mpi_rank == 0:
        if not os.path.isdir(os.path.dirname(fOut)):
            os.mkdir(os.path.dirname(fOut))
        else:
            if os.path.isfile(fOut):
                os.remove(fOut)

        data = Dataset(fOut, "w")

        for dim in ["tile", "nx", "ny", "nx1", "ny1"]:
            data.createDimension(dim, dimensions[dim])

        v0 = data.createVariable("lon", "f8", ("tile", "nx1", "ny1"))
        v0[:] = lon.data
        v0.units = units["coord"]
        v0 = data.createVariable("lat", "f8", ("tile", "nx1", "ny1"))
        v0[:] = lat.data
        v0.units = units["coord"]

        v1 = data.createVariable("lonA", "f8", ("tile", "nx", "ny"))
        v1[:] = lonA.data
        v1.units = units["coord"]
        v1 = data.createVariable("latA", "f8", ("tile", "nx", "ny"))
        v1[:] = latA.data
        v1.units = units["coord"]

        v2 = data.createVariable("dx", "f8", ("tile", "nx", "ny1"))
        v2[:] = dx.data
        v2.units = units["dist"]
        v2 = data.createVariable("dy", "f8", ("tile", "nx1", "ny"))
        v2[:] = dy.data
        v2.units = units["dist"]

        data.close()


def unstagger_coordinate(field: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Use: field =
            unstagger_coord(field, mode='mean')

    Unstaggers the coordinate that is +1 in length compared to the other.

    Inputs:
    - field (array): a staggered or unstaggered field
    - mode (str):
        - mean (average of boundaries),
        - first (first value only),
        - last (last value only)

    Outputs:
    - field - unstaggered

    ** currently only works with fields that are the same size in x-
    ** and y- directions.
    """

    fs = field.shape

    if len(fs) == 2:
        field = field[np.newaxis]
        zDim, dim1, dim2 = field.shape
    elif len(fs) == 3:
        zDim, dim1, dim2 = field.shape
    elif len(fs) == 4:
        zDim, dim1, dim2, dim3 = field.shape

    if mode == "mean":
        if dim1 > dim2:
            field = 0.5 * (field[:, 1:, :] + field[:, :-1, :])
        elif dim2 > dim1:
            field = 0.5 * (field[:, :, 1:] + field[:, :, :-1])
        elif dim1 == dim2:
            pass

    elif mode == "first":
        if dim1 > dim2:
            field = field[:, :-1, :]
        elif dim2 > dim1:
            field = field[:, :, :-1]
        elif dim1 == dim2:
            pass

    elif mode == "last":
        if dim1 > dim2:
            field = field[:, 1:, :]
        elif dim2 > dim1:
            field = field[:, :, 1:]
        elif dim1 == dim2:
            pass

    if len(fs) == 2:
        field = field[0]

    return field


def remap_winds_to_meteorological(
    u_input: np.ndarray, v_input: np.ndarray, nTiles: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use: zonal, meridional =
            remap_winds_to_meteorological(u, v, nTiles)

    Rotates winds to zonal and meridional based on
    tile position.

    Inputs:
    - u, v on C-grid or D-grid

    Outputs:
    - zonal, meridional winds on cell centers (A-grid)
        - zonal is west -> east
        - meridional is south -> north
    """
    zonal = np.zeros(u_input.shape)
    meridional = np.zeros(v_input.shape)

    for tile in range(nTiles):
        if tile in [0, 1, 5]:  # first two tiles are normal
            zonal[tile] = u_input[tile]
            meridional[tile] = v_input[tile]

        if tile == 2:
            zonal[tile] = v_input[tile]
            meridional[tile] = -u_input[tile]

        if tile in [3, 4]:
            zonal[tile] = -v_input[tile]
            meridional[tile] = u_input[tile]

    return zonal, meridional
