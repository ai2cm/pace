import copy as cp
import enum
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
from pace.dsl.stencil_config import CompilationConfig, RunMode
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
density = 1
layout = (1, 1)
nhalo = 3
pressure_base = 10
tracer_base = 1.0
nz_effective = 79
fvt_dict = {"grid_type": 0, "hord": 6}


class GridType(enum.Enum):
    AGrid = 1
    CGrid = 2
    DGrid = 3


def store_namelist_variables(local_variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use: namelist_dict =
            store_namelist_variables(local_variables)

    Stores namelist variables into a dictionary.

    Inputs:
    - locally defined variables (locals())

    Outputs:
    - namelist_dict (Dict)
    """

    namelist_variables = [
        "nx",
        "ny",
        "nz",
        "timestep",
        "nDays",
        "test_case",
        "plot_output_during",
        "plot_output_after",
        "plot_jupyter_animation",
        "figure_everyNhours",
        "write_initial_condition",
        "show_figures",
        "tracer_center",
        "nSeconds",
        "figure_everyNsteps",
        "nSteps_advection",
    ]

    namelist_dict = {}
    for i in namelist_variables:
        if i in local_variables:
            namelist_dict[i] = local_variables[i]

    return namelist_dict


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

    mpi_rank = mpi_comm.Get_rank()

    # to match fortran
    nx = namelistDict["nx"]
    ny = namelistDict["ny"]
    if "nz" in namelistDict.keys():
        nz = namelistDict["nz"]
    else:
        nz = 79

    dimensions = {
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "nx1": nx + 1,
        "ny1": ny + 1,
        "nhalo": nhalo,
        "nxhalo": nx + 2 * nhalo,
        "nyhalo": ny + 2 * nhalo,
        "tile": 6,
    }

    units = {
        "area": "m2",
        "coord": "degrees",
        "courant": "",
        "dist": "m",
        "tracer": "/",
        "pressure": "Pa",
        "psi": "kg/m/s",
        "wind": "m/s",
        "mass": "kg",
    }

    origins = {
        "halo": (0, 0),
        "halo_3d": (0, 0, 0),
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
        nx_tile=dimensions["nx"],
        ny_tile=dimensions["ny"],
        nz=nz_effective,
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

    compilation_config = CompilationConfig(
        backend=backend,
        rebuild=True,
        validate_args=True,
        format_source=False,
        device_sync=False,
        run_mode=RunMode.BuildAndRun,
        use_minimal_caching=False,
        communicator=communicator,
    )

    stencil_config = StencilConfig(
        compare_to_numpy=False,
        compilation_config=compilation_config,
        dace_config=dace_config,
    )

    grid_indexing = GridIndexing.from_sizer_and_communicator(
        sizer=sizer, cube=communicator
    )

    # set the domain so there is only one level in the vertical -- forced
    if not nz_effective == dimensions["nz"]:
        domain = grid_indexing.domain
        domain_new = list(domain)
        domain_new[2] = dimensions["nz"]
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
        data=configuration["metric_terms"].lon.data * 180 / np.pi,
        dims=("x_interface", "y_interface"),
        units=units["coord"],
        origin=origins["compute_2d"],
        extent=(dimensions["nx1"], dimensions["ny1"]),
        gt4py_backend=backend,
    )
    lat = Quantity(
        data=configuration["metric_terms"].lat.data * 180 / np.pi,
        dims=("x_interface", "y_interface"),
        units=units["coord"],
        origin=origins["compute_2d"],
        extent=(dimensions["nx1"], dimensions["ny1"]),
        gt4py_backend=backend,
    )

    if gather:
        lon = configuration["communicator"].gather(lon)
        lat = configuration["communicator"].gather(lat)

    return lon, lat


def create_initial_tracer(
    lon: Quantity,
    lat: Quantity,
    tracer: Quantity,
    center: Tuple[float, float] = (0.0, 0.0),
) -> Quantity:
    """
    Use: tracer =
            create_initial_tracer(lon, lat, metadata, tracer, target_tile)

    Calculates a gaussian-bell shaped multiplier for tracer initialization.
    It centers the bell at the longitude and latitude provided in center.

    Inputs:
    - lon and lat (in radians, and including halo points)
    - metadata (Dict)
    - tracer: empty array to be filled with tracer
    - center: (lon, lat) in degrees

    Outputs:
    - tracer: updated quantity
    """

    r0 = RADIUS / 3.0

    p_center = [np.deg2rad(center[0]), np.deg2rad(center[1])]


    for jj in range(tracer.data.shape[1]-1):
        for ii in range(tracer.data.shape[0]-1):

            p_dist = [lon.data[ii, jj], lat.data[ii, jj]]
            r = great_circle_distance_lon_lat(
                p_center[0], p_dist[0], p_center[1], p_dist[1], RADIUS, np
            )

            tracer.data[ii, jj, :] = (
                0.5 * (1.0 + np.cos(np.pi * r / r0)) if r < r0 else 0.0
            )

    return tracer


def calculate_streamfunction(
    lon_agrid: Quantity,
    lat_agrid: Quantity,
    lon: Quantity,
    lat: Quantity,
    psi: Quantity,
    psi_staggered: Quantity,
    test_case: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use: psi, psi_staggered =
            calculate_streamfunction(lon_agrid, lat_agrid, lon, lat, psi, psi_staggered, test_case)

    Creates streamfunction from input quantities, for defined test cases:
        - a) constant radius (as in Fortran test case 1)
        - b) radius varies with latitude, less spreading out.
        - c) south-north propagation from tile 1

    Inputs:
    - lonA, latA (Quantity): (in radians) on A-grid
    - lon, lat (Quantity): (in radians) on cell corners
    - psi, psi_staggered (empty quantities):
    - test_case ("a" or "b" or "c")

    Outputs:
    - psi (streamfunction)
    - psi_staggered (streamfunction on tile corners.)
    """

    yA_t = np.cos(lat_agrid.data) * np.sin(lon_agrid.data)
    zA_t = np.sin(lat_agrid.data)
    y_t = np.cos(lat.data) * np.sin(lon.data)
    z_t = np.sin(lat.data)

    if test_case == "a":
        RadA = RADIUS * np.ones(lon_agrid.data.shape)
        Rad = RADIUS * np.ones(lon.data.shape)
        multiplierA = zA_t
        multiplier = z_t
    elif test_case == "b":
        RadA = RADIUS * np.cos(lat_agrid.data / 2)
        Rad = RADIUS * np.cos(lat.data / 2)
        multiplierA = zA_t
        multiplier = z_t
    elif test_case == "c":
        RadA = RADIUS
        Rad = RADIUS
        multiplierA = -yA_t
        multiplier = -y_t
    elif test_case == "d":
        print("This case is in TESTING.")
        RadA = RADIUS * np.ones(lon.data.shape)
        Rad = RADIUS * np.ones(lon.data.shape)
        multiplierA = (yA_t + zA_t) / 1.5
        multiplier = (y_t + z_t) / 1.5
    else:
        RadA = np.ones(lon_agrid.data.shape) * np.nan
        Rad = np.ones(lon.data.shape) * np.nan
        multiplierA = np.nan
        multiplier = np.nan
        print("Please choose one of the defined test cases.")
        print("This will return gibberish.")

    Ubar = (2.0 * np.pi * RADIUS) / (12.0 * 86400.0)
    streamfunction = -1 * Ubar * RadA * multiplierA
    psi.data[:, :, :] = streamfunction[:, :, np.newaxis]
    streamfunction = -1 * Ubar * Rad * multiplier
    psi_staggered.data[:, :, :] = streamfunction[:, :, np.newaxis]

    return psi, psi_staggered


def calculate_winds_from_streamfunction_grid(
    psi: np.ndarray,
    dx: Quantity,
    dy: Quantity,
    u_grid: Quantity,
    v_grid: Quantity,
    grid: GridType = GridType.AGrid,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use: u_grid, v_grid =
            calculate_winds_from_streamfunction_grid(psi, dx, dy, u_Grid, v_grid, grid)


    Returns winds on a chosen grid based on streamfunction and grid spacing.

    Inputs:
    - psi: streamfunction
    - dx, dy: distance between points
    - u_grid, v_grid: empty quantities to be filled
    - grid: A, C, or D for winds to be returned on

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

    if grid == GridType.AGrid:
        if not (u_grid.metadata.dims == ("x", "y", "z") and v_grid.metadata.dims == ("x", "y", "z")):
            print('Incorrect wind input dimensions for A-grid.')
    elif grid == GridType.CGrid:
        if not (u_grid.metadata.dims == ("x", "y_interface", "z") and v_grid.metadata.dims == ("x_interface", "y", "z")):
            print('Incorrect wind input dimensions for C-grid.') 
    elif grid == GridType.DGrid:
        if not (u_grid.metadata.dims == ("x_interface", "y", "z") and v_grid.metadata.dims == ("x", "y_interface", "z")):
            print('Incorrect wind input dimensions for D-grid.')            

    if grid == GridType.AGrid:
        u_grid[:, 1:-1, :] = -0.5 * (psi.data[:, 2:, :] - psi.data[:, :-2, :]) / dy.data[:, 1:-1, np.newaxis]
        v_grid[1:-1, :, :] = 0.5 * (psi.data[2:, :, :] - psi.data[:-2, :, :]) / dx.data[1:-1, :, np.newaxis]

    elif grid == GridType.CGrid:
        u_grid.data[:, :-1, :] = -1 * (psi.data[:, 1:, :] - psi.data[:, :-1, :]) / dy.data[:, :-1, np.newaxis]
        v_grid.data[:-1, :, :] = (psi.data[1:, :, :] - psi.data[:-1, :, :]) / dx.data[:-1, :, np.newaxis]

    elif grid == GridType.DGrid:
        u_grid[:, 1:, :] = -(psi.data[:, 1:, :] - psi.data[:, :-1, :]) / dy.data[:, 1:, np.newaxis]
        v_grid[1:, :, :] = -(psi.data[1:, :, :] - psi.data[:-1, :, :]) / dy.data[1:, :, np.newaxis]

    return u_grid, v_grid


def create_initial_state(
    metric_terms: MetricTerms,
    metadata: Dict[str, Any],
    tracer_center: Tuple[float, float],
    test_case: str,
) -> Dict[str, Quantity]:
    """
    Use: initial_state =
            create_initial_state(metric_terms, quantity_factory, metadata, tracer_center,
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
    - metric_terms (cofiguration) from configure_domain()
    - metadata (Dict) from define_metadata()
    - tracerCenter
    - test_case ("a" or "b")

    Outputs:
    - initial_state (Dict):
        - delp
        - tracer
        - uC
        - vC
        - psi
    """

    dimensions, origins, units = split_metadata(metadata)

    lon_agrid = Quantity(
        data=metric_terms.lon_agrid.data,
        dims=("x", "y"),
        units=units["coord"],
        origin=origins["compute_2d"],
        extent=(dimensions["nx"], dimensions["ny"]),
        gt4py_backend=backend,
    )
    lat_agrid = Quantity(
        data=metric_terms.lat_agrid.data,
        dims=("x", "y"),
        units=units["coord"],
        origin=origins["compute_2d"],
        extent=(dimensions["nx"], dimensions["ny"]),
        gt4py_backend=backend,
    )

    lon = Quantity(
        data=metric_terms.lon.data,
        dims=("x_interface", "y_interface"),
        units=units["coord"],
        origin=origins["compute_2d"],
        extent=(dimensions["nx1"], dimensions["ny1"]),
        gt4py_backend=backend,
    )
    lat = Quantity(
        data=metric_terms.lat.data,
        dims=("x_interface", "y_interface"),
        units=units["coord"],
        origin=origins["compute_2d"],
        extent=(dimensions["nx1"], dimensions["ny1"]),
        gt4py_backend=backend,
    )

    # tracer
    tracer = Quantity(
        data=np.zeros((dimensions["nxhalo"]+1, dimensions["nyhalo"]+1, dimensions["nz"]+1)),
        dims=("x", "y", "z"),
        units=units["tracer"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx"], dimensions["ny"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    tracer = create_initial_tracer(
        lon_agrid,
        lat_agrid,
        tracer,
        tracer_center,
    )

    # # pressure
    delp = Quantity(
        data=np.ones((dimensions["nxhalo"]+1, dimensions["nyhalo"]+1, dimensions["nz"]+1)) * pressure_base,
        dims=("x", "y", "z"),
        units=units["pressure"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx"], dimensions["ny"], dimensions["nz"]),
        gt4py_backend=backend,
    )

    # # # streamfunction
    psi_agrid = Quantity(
        data=np.zeros((dimensions["nxhalo"]+1, dimensions["nyhalo"]+1, dimensions["nz"]+1)),
        dims=("x", "y", "z"),
        units=units["psi"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx"], dimensions["ny"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    psi = Quantity(
        data=np.zeros((dimensions["nxhalo"]+1, dimensions["nyhalo"]+1, dimensions["nz"]+1)),
        dims=("x_interface", "y_interface", "z"),
        units=units["psi"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx1"], dimensions["ny1"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    psi_agrid, psi = calculate_streamfunction(
        lon_agrid,
        lat_agrid,
        lon,
        lat,
        psi_agrid,
        psi,
        test_case,
    )

    # # winds
    dx = Quantity(
        data=metric_terms.dx.data,
        dims=("x", "y_interface"),
        units=units["dist"],
        origin=origins["compute_2d"],
        extent=(dimensions["nx"], dimensions["ny1"]),
        gt4py_backend=backend,
    )
    dy = Quantity(
        data=metric_terms.dy.data,
        dims=("x_interface", "y"),
        units=units["dist"],
        origin=origins["compute_2d"],
        extent=(dimensions["nx1"], dimensions["ny"]),
        gt4py_backend=backend,
    )

    u_cgrid = Quantity(
        data=np.zeros((dimensions["nxhalo"]+1, dimensions["nyhalo"]+1, dimensions["nz"]+1)),
        dims=("x", "y_interface", "z"),
        units=units["wind"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    v_cgrid = Quantity(
        data=np.zeros((dimensions["nxhalo"]+1, dimensions["nyhalo"]+1, dimensions["nz"]+1)),
        dims=("x_interface", "y", "z"),
        units=units["wind"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    grid = GridType.CGrid
    u_cgrid, v_cgrid = calculate_winds_from_streamfunction_grid(
        psi, dx, dy, u_cgrid, v_cgrid, grid=grid
    )

    initial_state = {
        "delp": delp,
        "tracer": tracer,
        "u_cgrid": u_cgrid,
        "v_cgrid": v_cgrid,
        "psi": psi_agrid,
    }

    return initial_state


def run_finite_volume_fluxprep(
    configuration: Dict[str, Any],
    initial_state: Dict[str, Quantity],
    metadata: Dict[str, Any],
    timestep: float,
) -> Dict[str, Quantity]:
    """
    Use: fluxPrep =
            run_finite_volume_fluxprep(configuration, initial_state,
            metadata, timestep)

    Initializes and runs the FiniteVolumeFluxPrep class to get
    initial states for mass flux, contravariant winds, area flux.

    Inputs:
    - configuration (Dict) from configure_domain()
    - initial_tate (Dict) from create_initial_state()
    - metadata (Dict) from define_metadata()
    - timestep (float) for advection

    Outputs:
    - flux_prep (Dict):
        - crx and cry
        - mfxd and mfyd
        - uc_contra, vc_contra
        - x_area_flux, y_area_flux
    """

    dimensions, origins, units = split_metadata(metadata)

    # create empty quantities to be filled
    empty = np.zeros(
        (dimensions["nxhalo"] + 1, dimensions["nyhalo"] + 1, dimensions["nz"]+1)
    )

    crx = Quantity(
        data=empty,
        dims=("x_interface", "y", "z"),
        units=units["courant"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    cry = Quantity(
        data=empty,
        dims=("x", "y_interface", "z"),
        units=units["courant"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    x_area_flux = Quantity(
        data=empty,
        dims=("x_interface", "y", "z"),
        units=units["area"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    y_area_flux = Quantity(
        data=empty,
        dims=("x", "y_interface", "z"),
        units=units["area"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    uc_contra = Quantity(
        data=empty,
        dims=("x_interface", "y", "z"),
        units=units["wind"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    vc_contra = Quantity(
        data=empty,
        dims=("x", "y_interface", "z"),
        units=units["wind"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        gt4py_backend=backend,
    )

    # intialize and run
    fvf_prep = FiniteVolumeFluxPrep(
        configuration["stencil_factory"], configuration["grid_data"]
    )

    fvf_prep(
        initial_state["u_cgrid"],
        initial_state["v_cgrid"],
        crx,
        cry,
        x_area_flux,
        y_area_flux,
        uc_contra,
        vc_contra,
        timestep,
    )  # THIS WILL MODIFY CREATED QUANTITIES, but not change uc, vc

    mfxd = Quantity(
        data=empty,
        dims=("x_interface", "y", "z"),
        units=units["mass"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx1"], dimensions["ny"], dimensions["nz"]),
        gt4py_backend=backend,
    )
    mfyd = Quantity(
        data=empty,
        dims=("x", "y_interface", "z"),
        units=units["mass"],
        origin=origins["compute_3d"],
        extent=(dimensions["nx"], dimensions["ny1"], dimensions["nz"]),
        gt4py_backend=backend,
    )

    mfxd.data[:] = x_area_flux.data[:] * initial_state["delp"].data[:] * density
    mfyd.data[:] = y_area_flux.data[:] * initial_state["delp"].data[:] * density

    flux_prep = {
        "crx": crx,
        "cry": cry,
        "mfxd": mfxd,
        "mfyd": mfyd,
        "uc_contra": uc_contra,
        "vc_contra": vc_contra,
        "x_area_flux": x_area_flux,
        "y_area_flux": y_area_flux,
    }

    return flux_prep


def build_tracer_advection(
    configuration: Dict[str, Any], tracers: Dict[str, Quantity]
) -> TracerAdvection:
    """
    Use: tracer_advection =
            build_tracer_advection(configuration, tracers)


    Initializes FiniteVolumeTransport and TracerAdvection classes.

    Inputs:
    - configuration (Dict) from configure_domain()
    - tracers (Dict) from initialState created by create_initial_state()

    Outputs:
    - tracer_advection - an instance of TracerAdvection class
    """

    fvtp_2d = FiniteVolumeTransport(
        configuration["stencil_factory"],
        configuration["grid_data"],
        configuration["damping_coefficients"],
        fvt_dict["grid_type"],
        fvt_dict["hord"],
    )

    tracer_advection = TracerAdvection(
        configuration["stencil_factory"],
        fvtp_2d,
        configuration["grid_data"],
        configuration["communicator"],
        tracers,
    )

    return tracer_advection


def prepare_everything_for_advection(
    configuration: Dict[str, Any],
    initial_state: Dict[str, Quantity],
    metadata: Dict[str, Any],
    timestep: float,
) -> Tuple[Dict[str, Any], TracerAdvection]:
    """
    Use: tracer_advection_data, tracer_advection =
        prepare_everything_for_advection(configuration, initial_state,
        metadata, timestep)

    Calls run_finite_volume_fluxprep() and build_tracer_advection().

    Inputs:
    - configuration from configure_domain()
    - initialState from create_initial_state()
    - metadata from define_metadata()
    - timestep (float) advection timestep

    Outputs:
    - tracer_advection_data (Dict):
        - tracers (Dict)
        - delp
        - mfxd and mfyd
        - crx and cry
    - tracer_advection: instance of TracerAdvection class

    """

    tracers = {"tracer": initial_state["tracer"]}

    flux_prep = run_finite_volume_fluxprep(
        configuration,
        initial_state,
        metadata,
        timestep,
    )

    tracer_advection = build_tracer_advection(configuration, tracers)

    tracer_advection_data = {
        "tracers": tracers,
        "delp": initial_state["delp"],
        "mfxd": flux_prep["mfxd"],
        "mfyd": flux_prep["mfyd"],
        "crx": flux_prep["crx"],
        "cry": flux_prep["cry"],
    }

    return tracer_advection_data, tracer_advection


def run_advection_step_with_reset(
    tracer_advection_data_initial: Dict[str, Quantity],
    tracer_advection_data: Dict[str, Quantity],
    tracer_advection: TracerAdvection,
    timestep,
) -> Dict[str, Quantity]:
    """
    Use: tracer_advection_data =
            run_advection_step_with_reset(tracer_advection_data_initial,
            tracer_advection_data,
            tracer_advection, timestep)


    Runs one timestep of tracer advection, which overwrites all
    but delp. Then resets mfxd, mfyd, crx, cry to their initial
    values. This is needed because tracAdv overwrites their
    values with 1/3 of the original, which leads to exponential
    decay of advection if left unchanged.

    Inputs:
    - tracer_advection_data_initial (Dict) - initial state data, to which
        all but tracer gets reset to after every step
    - tracer_advection_data(Dict) - data that gets updated during tracer advection
    - tracer_advection (TracerAdvection) instance of class
    - timestep (float)

    Outputs:
    - tracer_advection_data (Dict) with updated tracer only
    """

    tracer_advection(
        tracer_advection_data["tracers"],
        tracer_advection_data["delp"],
        tracer_advection_data["mfxd"],
        tracer_advection_data["mfyd"],
        tracer_advection_data["crx"],
        tracer_advection_data["cry"],
        timestep,
    )

    tracer_advection_data["delp"] = cp.deepcopy(tracer_advection_data_initial["delp"])
    tracer_advection_data["mfxd"] = cp.deepcopy(tracer_advection_data_initial["mfxd"])
    tracer_advection_data["mfyd"] = cp.deepcopy(tracer_advection_data_initial["mfyd"])
    tracer_advection_data["crx"] = cp.deepcopy(tracer_advection_data_initial["crx"])
    tracer_advection_data["cry"] = cp.deepcopy(tracer_advection_data_initial["cry"])

    return tracer_advection_data


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

        fig = plt.figure(figsize=(12, 6))
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

        nx = metadata["dimensions"]["nx"]
        ny = metadata["dimensions"]["ny"]
        ax.set_title(
            f"Cubed-sphere mesh with {nx} x {ny} cells per tile (c{nx})"
        )

        plt.savefig(fOut, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close("all")


def plot_projection_field(
    configuration: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    field: Union[Quantity, np.ndarray],
    plot_dict: Dict[str, Any],
    mpi_rank: Any,
    f_out: str,
    show: bool = False,
    unstagger: str = "first",
    level: int = 0
) -> None:
    """
    Use: plot_projection_field(configuration, metadata,
            field, plot_dict, mpi_rank, fOut,
            show=False)

    Creates a Robinson projection and plots provided field.

    Inputs:
    - configuration (Dict) from configure_domain()
    - metadata (Dict) from define_metadata()
    - field (Quantity or array) - single layer
    - plot_dict (Dict) - vmin, vmax, etc.
    - mpi_rank
    - f_out (str): file name to save to - if blank, not saved.
    - show (bool): whether to show image in notebook

    Outputs: saved figure
    """

    lon, lat = get_lon_lat_edges(configuration, metadata, gather=True)

    if mpi_rank == 0:

        if "vmin" not in plot_dict:
            plot_dict["vmin"] = 0
        if "vmax" not in plot_dict:
            plot_dict["vmax"] = 1
        if "units" not in plot_dict:
            plot_dict["units"] = "forgot to add units"
        if "title" not in plot_dict:
            plot_dict["title"] = "forgot to add title"

        if isinstance(field.data, np.ndarray):
            field = field.data
        
        if len(field.shape) == 4:
            field = field[:, :, :, level]

        field_plot = np.squeeze(field)
        field_plot = unstagger_coordinate(field_plot, unstagger)

        fig = plt.figure(figsize=(12, 6))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection=ccrs.Robinson())
        ax.set_facecolor(".4")

        f1 = pcolormesh_cube(
            lat.data,
            lon.data,
            field_plot,
            cmap=plot_dict["cmap"],
            vmin=plot_dict["vmin"],
            vmax=plot_dict["vmax"],
        )
        plt.colorbar(f1, label=plot_dict["units"])

        ax.set_title(plot_dict["title"])

        if len(f_out) > 0:
            plt.savefig(f_out, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close("all")


def plot_tracer_animation(
    configuration: Dict[str, Any],
    metadata: Dict[str, Dict[str, Any]],
    tracer_archive: List[Quantity],
    mpi_rank: int,
    plot_dict_tracer: Dict[str, Any],
    figure_everyNsteps: int,
    timestep: float,
    frames: Union[str, int] = "all",
) -> None:
    """
    Use: plot_tracer_animation(configuration, metadata,
            tracer_archive, mpi_rank, plot_dict_tracer,
            figure_everyNsteps, timestep, frames="all")

    Plots an interactive animation inside Jupyter notebook.

    Inputs:
    - configuration (Dict) from configure_domain()
    - metadata (Dict) from define_metadata()
    - tracer_archive (List) of tracer states
    - mpi_rank
    - plot_dict_tracer (Dict) of plotting settings
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
            plot_dict_tracer["title"] = "Tracer state @ hour: %.2f" % (
                (step * figure_everyNsteps * timestep) / 60 / 60
            )

            pcolormesh_cube(
                lat.data,
                lon.data,
                tracer_stack[step][:, :, :, 0],
                vmin=plot_dict_tracer["vmin"],
                vmax=plot_dict_tracer["vmax"],
                cmap=plot_dict_tracer["cmap"],
            )
            ax.set_title(plot_dict_tracer["title"])

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
        configuration["metric_terms"].lon_agrid.data * 180 / np.pi,
        ("x", "y"),
        units["coord"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny"]),
        backend,
    )
    latA = Quantity(
        configuration["metric_terms"].lat_agrid.data * 180 / np.pi,
        ("x", "y"),
        units["coord"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny"]),
        backend,
    )

    dx = Quantity(
        configuration["metric_terms"].dx.data,
        ("x", "y_interface"),
        units["dist"],
        origins["compute_2d"],
        (dimensions["nx"], dimensions["ny1"]),
        backend,
    )
    dy = Quantity(
        configuration["metric_terms"].dy.data,
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
