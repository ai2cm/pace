import copy as cp
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


backend = "numpy"
layout = (1, 1)
fvt_dict = {"grid_type": 0, "hord": 6}


def split_metadata(
    metadata: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, int], Dict[str, Tuple[Any]], Dict[str, str]]:
    """
    Use: dimensions, origins, units = split_metadata(metadata)

    metadata is output of define_metadata().

    Inputs:
    - metadata (Dict)

    Outputs:
    - dimensions (Dict)
    - origins (Dict)
    - units (Dict)
    """

    dimensions = metadata["dimensions"]
    origins = metadata["origins"]
    units = metadata["units"]

    return dimensions, origins, units


def store_namelist_variables(local_variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use: namelistDict = store_namelist_variables(local_variables)

    Inputs:
    - localVariables: namespace

    Outputs:
    - namelistDict (Dict)
    """

    namelist_variables = [
        "nx",
        "ny",
        "nhalo",
        "timestep",
        "nDays",
        "test_case",
        "print_advectionProgress",
        "plot_outputDuring",
        "plot_outputAfter",
        "plot_jupyterAnimation",
        "figure_everyNhours",
        "write_initialCondition",
        "plot_gridLayout",
        "show_figures",
        "pressure_base",
        "tracer_base",
        "tracer_target_tile",
        "density",
        "nSeconds",
        "figure_everyNsteps",
        "nSteps_advection",
    ]

    namelistDict = {}
    for i in namelist_variables:
        namelistDict[i] = local_variables[i]

    return namelistDict


def get_lon_lat_edges(
    configuration: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    gather: bool = True,
) -> Tuple[Quantity, Quantity]:
    """
    Use: lon, lat = get_lon_lat_edges(
        configuration, metadata, gather=True)

    Creates lon and lat Quantities with data in degrees.

    Inputs:
    - configuration (Dict)
    - metadata (Dict)
    - gather (bool): - gathers on rank 0

    Outputs:
    - lon and lat on A-grid in degrees
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


def define_metadata(
    namelistDict: Dict[str, Any], mpi_comm: Any
) -> Tuple[Dict[str, int], Dict[str, Tuple[Any]], Dict[str, str], int]:
    """
    Use: metadata, mpi_rank =
    define_metadata(namelistDict, mpi_comm)

    Outputs dictionaries for basic configuration parameters.

    Inputs:
    - namelistDict (Dict)
    - mpi_comm: mpi communicator

    Outputs:
    - dimensions (Dict)
    - units (Dict)
    - origins (Dict)
    - mpi_rank: rank for each of the subprocesses
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
        "nhalo": namelistDict["nhalo"],
        "nxhalo": nx + 2 * namelistDict["nhalo"],
        "nyhalo": ny + 2 * namelistDict["nhalo"],
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


def configure_domain(mpi_comm: Any, dimensions: Dict[str, int]) -> Dict[str, Any]:
    """
    Use: configuration = configure_domain(mpi_comm, dimensions)

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


def create_gaussian_multiplier(
    lon: np.ndarray,
    lat: np.ndarray,
    dimensions: Dict[str, int],
    mpi_rank: Any,
    target_tile: int = 0,
) -> np.ndarray:
    """
    Use: gaussian_multiplier =
    create_gaussian_multiplier(lon, lat, dimensions, mpi_rank, target_tile=0)

    Creates a 2-D Gaussian bell shape on the desired tile/rank in the domain.

    Inputs:
    - lon, lat: longitude and latitude of centerpoints (A-grid) (in radians)
    - dimensions (Dict)
    - mpi_rank
    - target_tile: the tile on which to center the blob

    Outputs:
    - gaussian_multiplier (array): blob centered at the middle of target_tile
        with gaussian dropoff
    """

    r0 = RADIUS / 3.0
    p1x, p1y = dimensions["nxhalo"] // 2, dimensions["nyhalo"] // 2  # center gaussian on middle of tile
    gaussian_multiplier = np.zeros((dimensions["nxhalo"], dimensions["nyhalo"]))

    if mpi_rank == target_tile:
        p_center = [lon[p1x, p1y], lat[p1x, p1y]]
        print(
            "Centering gaussian on lon=%.2f, lat=%.2f"
            % (np.rad2deg(p_center[0]), np.rad2deg(p_center[1]))
        )

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


def calculate_streamfunction_testcase1a(
    lon: np.ndarray, lat: np.ndarray, dimensions: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use: psi, psi_staggered = calculate_streamfunction_testcase1a(lon, lat, dimensions)

    Calculates streamfunction for testCase1 in fortran. Runs on each rank independently.
    Streamfunction is calculated by multiplying -wind (-Ubar), the radius of the earth 
    (RADIUS), and sin(latitude), so it decreases from the South pole to the North pole.

    Inputs:
    - lon: longitude of center points (in radians)
    - lat: latitude of center points (in radians)
    - dimensions (Dict)

    Outputs:
    - psi: streamfunction on tile centers (with halo points)
    - psi_staggered: streamfunction on tile corners (with halo points)
    """

    Ubar = (2.0 * np.pi * RADIUS) / (12.0 * 86400.0)  # 38.6
    alpha = 0

    psi = np.ones((dimensions["nxhalo"], dimensions["nyhalo"])) * 1.0e25
    psi_staggered = np.ones((dimensions["nxhalo"], dimensions["nyhalo"])) * 1.0e25

    for jj in range(dimensions["nyhalo"]):
        for ii in range(dimensions["nxhalo"]):
            psi[ii, jj] = (
                -1.0
                * Ubar
                * RADIUS
                * (
                    np.sin(lat[ii, jj]) * np.cos(alpha)
                    - np.cos(lon[ii, jj]) * np.cos(lat[ii, jj]) * np.sin(alpha)
                )
            )

    for jj in range(dimensions["nyhalo"]):
        for ii in range(dimensions["nxhalo"]):
            psi_staggered[ii, jj] = (
                -1.0
                * Ubar
                * RADIUS
                * (
                    np.sin(lat[ii, jj]) * np.cos(alpha)
                    - np.cos(lon[ii, jj]) * np.cos(lat[ii, jj]) * np.sin(alpha)
                )
            )

    return psi, psi_staggered


def calculate_streamfunction_testcase1b(
    lon, lat, dimensions: Dict[str, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use: psi, psi_staggered = calculate_streamfunction_testcase1b(lon, lat, dimensions)

    Calculates streamfunction for testCase1 in fortran. Runs on each rank independently.
    Modified Ubar so it depends on radius (same period regardless of latitude?)
    Some experimentation here.

    Inputs:
    - lon: longitude of center points (in radians)
    - lat: latitude of center points (in radians)
    - dimensions (Dict)

    Outputs:
    - psi: streamfunction on tile centers (with halo points)
    - psi_staggered: streamfunction on tile corners (with halo points)
    """

    R_lat = RADIUS * np.cos(lat / 2)
    Ubar = (2.0 * np.pi * RADIUS) / (12.0 * 86400.0)
    alpha = 0

    psi = np.ones((dimensions["nxhalo"], dimensions["nyhalo"])) * 1.0e25
    psi_staggered = np.ones((dimensions["nxhalo"], dimensions["nyhalo"])) * 1.0e25

    for jj in range(dimensions["nyhalo"]):
        for ii in range(dimensions["nxhalo"]):
            psi[ii, jj] = (
                -1.0
                * Ubar
                * R_lat[ii, jj]
                * (
                    np.sin(lat[ii, jj]) * np.cos(alpha)
                    - np.cos(lon[ii, jj]) * np.cos(lat[ii, jj]) * np.sin(alpha)
                )
            )

    for jj in range(dimensions["nyhalo"]):
        for ii in range(dimensions["nxhalo"]):
            psi_staggered[ii, jj] = (
                -1.0
                * Ubar
                * R_lat[ii, jj]
                * (
                    np.sin(lat[ii, jj]) * np.cos(alpha)
                    - np.cos(lon[ii, jj]) * np.cos(lat[ii, jj]) * np.sin(alpha)
                )
            )

    return psi, psi_staggered


def calculate_winds_from_streamfunction_grid(
    psi: Quantity,
    dx: Quantity,
    dy: Quantity,
    dimensions: Dict[str, int],
    grid: str = "A",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use: u_grid, v_grid =
    calculate_winds_from_streamfunction_grid(psi, dx, dy, dimensions, grid="A")

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
    - C: streamfunction on corder points, dx, dy on edge points, all with halos
    - D: streamfunction on center points, dx, dy on c-grid, all with halos
    """
    u_grid = np.zeros((dimensions["nxhalo"], dimensions["nyhalo"]))
    v_grid = np.zeros((dimensions["nxhalo"], dimensions["nyhalo"]))

    if grid == "A":
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

    elif grid == "C":
        for jj in range(
            dimensions["nhalo"] - 1, dimensions["ny"] + dimensions["nhalo"] + 1
        ):
            for ii in range(
                dimensions["nhalo"] - 1, dimensions["nx"] + dimensions["nhalo"] + 1
            ):
                dist = dx.data[ii, jj]
                v_grid[ii, jj] = (
                    0 if dist == 0 else (psi.data[ii + 1, jj] - psi.data[ii, jj]) / dist
                )

                dist = dy.data[ii, jj]
                u_grid[ii, jj] = (
                    0
                    if dist == 0
                    else -1.0 * (psi.data[ii, jj + 1] - psi.data[ii, jj]) / dist
                )

    elif grid == "D":
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


def create_initialstate_testcase1a(
    grid_data: GridData,
    metadata: Dict[str, Dict[str, Any]],
    tracer_dict: Dict[str, str],
    pressure_base: int,
) -> Dict[str, Quantity]:
    """
    Use: initialState =
    create_initialstate_testcase1a(grid_data, metadata, tracer_dict, pressure_base)

    Creates inital state from the fortran test_case 1 streamfunction configuration -
    pressure, gaussian tracer distribution, u and v winds on C-grid.

    Inputs:
    - grid_data: configuration["grid_data"]
    - metadata (Dict):
        - dimensions: Dict{"nx", "ny", "nx1", "ny1", "nxhalo", "nyhalo", "tile"}
        - units: Dict{"coord", "dist", "mass", "psi", "wind"}
        - origins: Dict{"halo", "compute_2d"}
    - tracer_dict: Dict{"target_tile", "rank", "tracer_base"}
    - pressure_base: pressure thickess of layer in Pa

    Outputs:
    - initialState: Dict{"delp", "tracer", "uc", "vc"} - 3D with a
        single layer in the vertical
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

    # tracer
    gaussian_multiplier = create_gaussian_multiplier(
        lonA_halo.data,
        latA_halo.data,
        dimensions,
        target_tile=tracer_dict["target_tile"],
        mpi_rank=tracer_dict["rank"],
    )
    tracer = Quantity(
        gaussian_multiplier * tracer_dict["tracer_base"],
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
    _, psi_staggered = calculate_streamfunction_testcase1a(
        lonA_halo.data, latA_halo.data, dimensions
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
    uC, vC = calculate_winds_from_streamfunction_grid(
        psi_staggered_halo, dx_halo, dy_halo, dimensions, grid="C"
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


def create_initialstate_testcase1b(
    grid_data: GridData,
    metadata: Dict[str, Dict[str, Any]],
    tracer_dict: Dict[str, str],
    pressure_base: int,
) -> Dict[str, Quantity]:
    """
    Use: initialState =
    create_initialstate_testcase1b(grid_data, metadata, tracer_dict, pressure_base)

    Creates inital state from the fortran test_case 1 streamfunction configuration -
    pressure, gaussian tracer distribution, u and v winds on C-grid.
    Streamfunction modified to have weaker winds away from equator

    Inputs:
    - grid_data: configuration["grid_data"]
    - metadata (Dict):
        - dimensions: Dict{"nx", "ny", "nx1", "ny1", "nxhalo", "nyhalo", "tile"}
        - units: Dict{"coord", "dist", "mass", "psi", "wind"}
        - origins: Dict{"halo", "compute_2d"}
    - tracer_dict: Dict{"target_tile", "rank", "tracer_base"}
    - pressure_base: pressure thickess of layer in Pa

    Outputs:
    - initialState: Dict{"delp", "tracer", "uc", "vc"} - 3D with a
        single layer in the vertical
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

    # TRACER
    gaussian_multiplier = create_gaussian_multiplier(
        lonA_halo.data,
        latA_halo.data,
        dimensions,
        target_tile=tracer_dict["target_tile"],
        mpi_rank=tracer_dict["rank"],
    )
    tracer = Quantity(
        gaussian_multiplier * tracer_dict["tracer_base"],
        ("x", "y"),
        units["mass"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny"]),
        backend,
    )

    # PRESSURE
    delp = Quantity(
        np.ones(tracer.data.shape) * pressure_base,
        ("x", "y"),
        units["pressure"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny"]),
        backend,
    )

    # STREAMFUNCTION
    _, psi_staggered = calculate_streamfunction_testcase1b(
        lonA_halo.data, latA_halo.data, dimensions
    )
    psi_staggered_halo = Quantity(
        psi_staggered,
        ("x_halo", "y_halo"),
        units["psi"],
        origins["halo"],
        (dimensions["nxhalo"], dimensions["nyhalo"]),
        backend,
    )

    # WINDS
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
    uC, vC = calculate_winds_from_streamfunction_grid(
        psi_staggered_halo, dx_halo, dy_halo, dimensions, grid="C"
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

    # EXTEND INITIAL CONDITIONS INTO ONE VERTICAL LAYER
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
    metadata: Dict[str, Dict[str, Any]],
    density: float,
    timestep: float,
) -> Dict[str, Quantity]:
    """
    Use: fluxPrep =
    run_finite_volume_fluxprep(configuration, initialState, metadata, density, timestep)

    Initializes FiniteVolumeFluxPrep class and fills in variables
    from initial wind data.

    Inputs:
    - configuration (Dict): "stencil_factory", "grid_data"
    - initialState (Dict): "uC", "vC", "delp"
    - metadata (Dict):=
        - dimensions: Dict{"nx", "ny", "nz", "nx1", "ny1", "nxhalo", "nyhalo"}
        - units: Dict{"areaflux", "courant", "mass", "wind"}
        - origins: Dict{"compute_3d"}
    - density (float): air density (kg/m3)
    - timestep (float): advection time step

    Outputs:
    - fluxPrep: Dict{"crx", "cry", "mfxd", "mfyd", "ucv", "vcv", "xaf", "yaf"}
        - crx, cry are Courant numbers
        - ucv, vcv are the contravariant winds
        - xaf, yaf are fluxes of area (dx*dy but incorporating wind speed)
        - mfxd, mfyd are mass (volume) fluxes incorporating wind speed
    """

    uc = initialState["uC"]
    vc = initialState["vC"]
    delp = initialState["delp"]

    dimensions, origins, units = split_metadata(metadata)

    # CREATE EMPTY QUANTITIES TO BE FILLED
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

    # INITIALIZE AND RUN
    fvf_prep = FiniteVolumeFluxPrep(
        configuration["stencil_factory"], configuration["grid_data"]
    )

    fvf_prep(
        uc, vc, crx, cry, xaf, yaf, ucv, vcv, timestep
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

    mfxd.data[:] = xaf.data[:] * delp.data[:] * density
    mfyd.data[:] = yaf.data[:] * delp.data[:] * density

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
    configuration: Dict[str, Any],
    fvt_dict: Dict[str, int],
    tracers: Dict[str, Quantity],
) -> TracerAdvection:
    """
    Use: tracAdv = build_tracer_advection(configuration, fvt_dict, tracers)

    Initializes the tracer advection class from FiniteVolumeTransport
    and TracerAdvection.

    Inputs:
    - configuration: Dict{"stencil_factory", "grid_data",
                          "damping_coefficients"}
    - fvt_dict: Dict{"grid_type", "hord"}
    - tracers: Dict{"tracer}

    Outputs:
    - tracAdv: class instance that performs tracer advection
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
    metadata: Dict[str, Dict[str, Any]],
    density: float,
    timestep: float,
) -> Tuple[Dict[str, Quantity], TracerAdvection]:
    """
    Use: fluxPrep, tracAdv = prepare_everything_for_advection(
        configuration, initialState, metadata, density, timestep, fvt_dict)

    Inputs:
    - configuration (Dict): "stencil_factory", "grid_data", "damping_coefficients"
    - initialState (Dict): "uC", "vC", "delp", "tracer"
    - metadata (Dict):
        - dimensions: Dict{"nx", "ny", "nz", "nx1", "ny1", "nxhalo", "nyhalo"}
        - units: Dict{"areaflux", "courant", "mass", "wind"}
        - origins: Dict{"compute_3d"}
    - density: air density in kg/m3
    - timestep: advection time step

    Outputs:
    - fluxPrep: Dict{"crx", "cry", "mfxd", "mfyd", "ucv", "vcv", "xaf", "yaf"}
        - crx, cry are Courant numbers
        - ucv, vcv are the contravariant winds
        - xaf, yaf are fluxes of area (dx*dy but incorporating wind speed)
        - mfxd, mfyd are mass (volume) fluxes incorporating wind speed
    - tracAdv: class instance that performs tracer advection
    """

    delp = initialState["delp"]
    tracers = {"tracer": initialState["tracer"]}

    fluxPrep = run_finite_volume_fluxprep(
        configuration, initialState, metadata, density, timestep
    )

    tracAdv = build_tracer_advection(configuration, fvt_dict, tracers)

    tracAdv_data = {
        "tracers": tracers,
        "delp": delp,
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
    timestep: float,
    mpi_rank: int = None,
) -> Dict[str, Quantity]:
    """
    Use: tracAdv_data = run_advection_step_with_reset()

    Inputs:
    - tracAdv_dataInit: Dict{"tracers", "delp", "mfxd", "mfyd", "crx", "cry"}
                        of values to be reset to after each advection step
    - tracAdv_data: Dict{"tracers", "delp", "mfxd", "mfyd", "crx", "cry"}
    - tracAdv: class instance of TracerAdvection
    - timestep: time step in seconds
    - mpi_rank: if 0, prints differences from timestep, initial condition

    Outputs:
    - tracAdv_data: updated fields (of only tracer)
    """

    preAdvection_tracerState = cp.deepcopy(tracAdv_data["tracers"])

    tracAdv(
        tracAdv_data["tracers"],
        tracAdv_data["delp"],
        tracAdv_data["mfxd"],
        tracAdv_data["mfyd"],
        tracAdv_data["crx"],
        tracAdv_data["cry"],
        timestep,
    )

    tracAdv_data["mfxd"] = cp.deepcopy(tracAdv_dataInit["mfxd"])
    tracAdv_data["mfyd"] = cp.deepcopy(tracAdv_dataInit["mfyd"])
    tracAdv_data["crx"] = cp.deepcopy(tracAdv_dataInit["crx"])
    tracAdv_data["cry"] = cp.deepcopy(tracAdv_dataInit["cry"])

    # print diagnostics if mpi_rank is provided
    if mpi_rank == 0:
        diff_timestep = tracAdv_data["tracers"]["tracer"].data - preAdvection_tracerState["tracer"].data
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


def plot_projection_field(
    configuration: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    field: Quantity,
    plotDict: Dict[str, Any],
    mpi_rank: Any,
    fOut: str,
    show: bool = False,
):

    # mpi_rank, show=False):
    """
    Use: plot_projection_field
    (configuration, metadata, field, plotDict, mpi_rankm fOut, show=False)

    Creates a Robinson projection and plots the (6) tiles of field on a map.

    Inputs:
    - configuration (Dict): "grid_data", "communicator"
    - medatada (Dict): "dimensions", "origins", "units"
    - field: unstaggered field at a given vertical level (tile, x, y)
    - plotDict_tracer: Dict{"vmin", "vmax", "units", "title", "cmap"}
    - mpi_rank
    - fOut: save image to this file
    - show (bool): whether to draw the image in jupyter notebook

    Outputs: plots figure
    """

    lon, lat = get_lon_lat_edges(configuration, metadata, gather=True)

    if mpi_rank == 0:
        if not os.path.isdir(os.path.dirname(fOut)):
            os.mkdir(os.path.dirname(fOut))
        else:
            if os.path.isfile(fOut):
                os.remove(fOut)

        field_plot = field.data[:, :, :, 0]

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



def plot_grid(
    configuration: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    mpi_rank: Any,
    fSave: str = "grid_map.png",
    show: bool = False,
):
    """
    Use: plot_grid(
        configuration, metadata, fSave="grid_map.png", show=False)

    Creates a Robinson projection and plots grid edges.

    Inputs:
    - configuration (Dict): "communicator", "grid_data"
    - metadata (Dict): "dimensions", "origins", "units"
    - fSave (str): file name
    - show (Bool): whether to show image in notebook

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

        plt.savefig(fSave, dpi=300, bbox_inches="tight")
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
):
    """
    Use:
    plot_tracer_animation(
        configuration, metadata, tracer_archive, mpi_rank,
        plotDict_tracer, figure_everyNsteps, timestep, frames="all"
        )

    Creates an interactive animation inside jupyter notebook.
    (takes about 3.5 seconds per frame)

    Inputs:
    - configuration (Dict): "grid_data", "communicator"
    - metadata (Dict)
    - tracer_archive (List): list of stored tracer states
    - mpi_rank
    - plotDict_tracer (Dict): plotting parameters
    - figure_everyNsteps (Int): how many steps to skip between frames
    - timestep
    - frames ("all" or Int): how many frames of the animation to plot

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

        def animate(step):
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
    variables: Dict[str, Quantity],
    metadata: Dict[str, Dict[str, Any]],
    configuration: Dict[str, Any],
    mpi_rank: int,
):
    """
    Use: write_initial_condition_tofile(
        fOut, variables, metadata, configuration, mpi_rank)

    Creates netCDF file with initial conditions.

    Inputs:
    - fOut: output netcdf file
    - variables: initialState
    - metadata:
        - dimensions: Dict{"tile", "nx", "ny", "nx1", "ny1"}
        - units: Dict{"coord", "wind", "pressure", "mass"}
    - configuration (Dict): "communicator", "grid_data"
    - mpi_rank

    Outputs: written netCDF file
    """

    dimensions, _, units = split_metadata(metadata)

    lon, lat = get_lon_lat_edges(configuration, metadata, gather=True)

    uC = np.squeeze(configuration["communicator"].gather(variables["uC"]))
    vC = np.squeeze(configuration["communicator"].gather(variables["vC"]))
    delp = np.squeeze(configuration["communicator"].gather(variables["delp"]))
    tracer = np.squeeze(configuration["communicator"].gather(variables["tracer"]))

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

        v1 = data.createVariable("delp", "f8", ("tile", "nx", "ny"))
        v1[:] = delp.data
        v1.units = units["pressure"]

        v1 = data.createVariable("tracer", "f8", ("tile", "nx", "ny"))
        v1[:] = tracer.data
        v1.units = units["mass"]

        data.close()



def unstagger_coordinate(field: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Use: field = unstagger_coord(field, mode="mean")

    Unstaggers the coordinate that is +1 in length compared to the other.

    Inputs:
    - field: a staggered or unstaggered field
    - mode: mean (average of boundaries), first (first value only),
            last (last value only)

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
    
    else:
        print('Mode not supported.')
        field = np.nan

    if len(fs) == 2:
        field = field[0]

    return field
