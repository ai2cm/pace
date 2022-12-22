import copy as cp
import enum
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube
from IPython.display import HTML, display
from matplotlib import animation
from units_config import units

from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.stencil import GridIndexing, StencilConfig, StencilFactory
from pace.dsl.stencil_config import CompilationConfig, RunMode
from pace.fv3core.stencils.fvtp2d import FiniteVolumeTransport
from pace.fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from pace.fv3core.stencils.tracer_2d_1l import TracerAdvection
from pace.util import (
    CubedSphereCommunicator,
    CubedSpherePartitioner,
    Quantity,
    QuantityFactory,
    SubtileGridSizer,
    TilePartitioner,
)
from pace.util.constants import RADIUS
from pace.util.grid import (
    AngleGridData,
    ContravariantGridData,
    DampingCoefficients,
    GridData,
    HorizontalGridData,
    MetricTerms,
    VerticalGridData,
)
from pace.util.grid.gnomonic import great_circle_distance_lon_lat


class GridType(enum.Enum):
    AGrid = 1
    CGrid = 2
    DGrid = 3


class VariableDims(enum.Enum):
    XY = 1
    XYZ = 2


class VariableGrid(enum.Enum):
    CellCenters = 1
    CellCorners = 2
    StaggeredInX = 3
    StaggeredInY = 4


def init_quantity(
    dimensions: Dict,
    grid: VariableGrid,
    dims: VariableDims = VariableDims.XYZ,
    units: str = "",
    backend: str = "numpy",
) -> Quantity:
    """
    Use: output = init_quantity(dimensions, grid, dims, units)

    Creates a zero-filled quantity (either 2- or 3-d) with correct
    dims, extent, origin, etc.

    Inputs:
    - dimensions (Dict)
    - grid (VariableGrid):
        - CellCenters,
        - CellCorners,
        - StaggeredInX,
        - StaggeredInY
    - dims (VariableDims):
        - XY (2-dimensional, like lat, lon, dx, etc.)
        - XYZ (3-dimensional)

    Outputs:
    - output (Quantity) with set metadata
    """

    nx, ny, nz = dimensions["nx"], dimensions["ny"], dimensions["nz"]
    nhalo = dimensions["nhalo"]

    nx_all, ny_all = nx + 2 * nhalo + 1, ny + 2 * nhalo + 1
    nz_all = nz + 1

    if dims == VariableDims.XY:
        empty = np.zeros((nx_all, ny_all))
        skip_z = -1
    elif dims == VariableDims.XYZ:
        empty = np.zeros((nx_all, ny_all, nz_all))
        skip_z = None

    variable = 0
    if grid == VariableGrid.CellCenters:
        variable = Quantity(
            data=empty,
            dims=("x", "y", "z")[:skip_z],
            units=units,
            origin=(nhalo, nhalo, 0)[:skip_z],
            extent=(nx, ny, nz)[:skip_z],
            gt4py_backend=backend,
        )

    if grid == VariableGrid.CellCorners:
        variable = Quantity(
            data=empty,
            dims=("x_interface", "y_interface", "z")[:skip_z],
            units=units,
            origin=(nhalo, nhalo, 0)[:skip_z],
            extent=(nx + 1, ny + 1, nz)[:skip_z],
            gt4py_backend=backend,
        )

    elif grid == VariableGrid.StaggeredInX:
        variable = Quantity(
            data=empty,
            dims=("x_interface", "y", "z")[:skip_z],
            units=units,
            origin=(nhalo, nhalo, 0)[:skip_z],
            extent=(nx + 1, ny, nz)[:skip_z],
            gt4py_backend=backend,
        )

    elif grid == VariableGrid.StaggeredInY:
        variable = Quantity(
            data=empty,
            dims=("x", "y_interface", "z")[:skip_z],
            units=units,
            origin=(nhalo, nhalo, 0)[:skip_z],
            extent=(nx, ny + 1, nz)[:skip_z],
            gt4py_backend=backend,
        )

    return variable


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
        "nhalo",
        "layout",
        "timestep",
        "nDays",
        "test_case",
        "plot_output_after",
        "plot_jupyter_animation",
        "figure_everyNhours",
        "write_initial_condition",
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


def define_dimensions(
    namelistDict: Dict[str, Any]
) -> Tuple[Dict[str, int], Dict[str, Tuple[Any]], Dict[str, str], int]:
    """
    Use: dimensions =
            define_dimensions(namelistDict)

    Creates dictionary that stores dimension information.

    Inputs:
    - namelistDict (Dict) from store_namelist_variables()

    Outputs:
    - dimensions (Dict)
    """

    # to match fortran
    nhalo = namelistDict["nhalo"]
    nx = namelistDict["nx"]
    ny = namelistDict["ny"]
    layout = namelistDict["layout"]
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
        "layout": layout,
    }
    return dimensions


def configure_domain(
    mpi_comm: Any,
    dimensions: Dict[str, int],
    single_layer: bool = True,
    backend: str = "numpy",
) -> Dict[str, Any]:
    """
    Use: configuration =
            configure_domain(mpi_comm, dimensions)

    Creates all domain configuration parameters and stores them.

    Inputs:
    - mpi_comm: communicator
    - dimensions (Dict)

    Outputs:
    - domain_configuration (Dict):
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

    partitioner = CubedSpherePartitioner(TilePartitioner(dimensions["layout"]))
    communicator = CubedSphereCommunicator(mpi_comm, partitioner)

    sizer = SubtileGridSizer.from_tile_params(
        nx_tile=dimensions["nx"],
        ny_tile=dimensions["ny"],
        nz=dimensions["nz"],
        n_halo=dimensions["nhalo"],
        extra_dim_lengths={},
        layout=dimensions["layout"],
        tile_partitioner=partitioner.tile,
        tile_rank=communicator.tile.rank,
    )

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)

    metric_terms = MetricTerms(
        quantity_factory=quantity_factory, communicator=communicator
    )

    # workaround for single layer
    if single_layer:
        horizontal_grid_data = HorizontalGridData.new_from_metric_terms(metric_terms)
        vertical_grid_data = VerticalGridData(ak=[10.0, 0.0], bk=[0.0, 1.0])
        contravariant_grid_data = ContravariantGridData.new_from_metric_terms(
            metric_terms
        )
        angle_grid_data = AngleGridData.new_from_metric_terms(metric_terms)

        grid_data = GridData(
            horizontal_grid_data,
            vertical_grid_data,
            contravariant_grid_data,
            angle_grid_data,
        )

    else:
        grid_data = GridData.new_from_metric_terms(metric_terms)

    domain_configuration = {
        "partitioner": partitioner,
        "communicator": communicator,
        "sizer": sizer,
        "quantity_factory": quantity_factory,
        "metric_terms": metric_terms,
        "grid_data": grid_data,
    }

    return domain_configuration


def configure_stencil(
    domain_configuration: Dict[str, Any],
    backend: str = "numpy",
    single_layer: bool = True,
) -> Dict[str, Any]:
    """
    Use:
    stencil_configuration = configure_stencil(
        domain_configuration, backend="numpy", single_layer=True)

    Inputs:
    - domain configuration (Dict) from configure_domain()
    - backend (only works for numpy)
    - single_layer - should be true if nz=1

    Outputs:
    - stencil_configuraton (Dict):
        - grid_data
        - damping_coefficients
        - dace_config
        - compilation_config
        - stencil_config
        - grid_indexing
        - stencil_factory
    """

    metric_terms = domain_configuration["metric_terms"]

    if single_layer:
        horizontal_grid_data = HorizontalGridData.new_from_metric_terms(metric_terms)
        vertical_grid_data = VerticalGridData(ak=[10], bk=[0])
        contravariant_grid_data = ContravariantGridData.new_from_metric_terms(
            metric_terms
        )
        angle_grid_data = AngleGridData.new_from_metric_terms(metric_terms)

        grid_data = GridData(
            horizontal_grid_data,
            vertical_grid_data,
            contravariant_grid_data,
            angle_grid_data,
        )

    else:
        grid_data = GridData.new_from_metric_terms(metric_terms)

    damping_coefficients = DampingCoefficients.new_from_metric_terms(
        domain_configuration["metric_terms"]
    )

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
        communicator=domain_configuration["communicator"],
    )

    stencil_config = StencilConfig(
        compare_to_numpy=False,
        compilation_config=compilation_config,
        dace_config=dace_config,
    )

    grid_indexing = GridIndexing.from_sizer_and_communicator(
        sizer=domain_configuration["sizer"], cube=domain_configuration["communicator"]
    )

    stencil_factory = StencilFactory(config=stencil_config, grid_indexing=grid_indexing)

    stencil_configuration = {
        "grid_data": grid_data,
        "communicator": domain_configuration["communicator"],
        "damping_coefficients": damping_coefficients,
        "dace_config": dace_config,
        "stencil_config": stencil_config,
        "grid_indexing": grid_indexing,
        "stencil_factory": stencil_factory,
    }

    return stencil_configuration


def get_lon_lat_edges(
    domain_configuration: Dict[str, Any],
    dimensions: Dict[str, int],
    gather: bool = True,
) -> Tuple[Quantity, Quantity]:
    """
    Use: lon, lat =
    get_lon_lat_edges(configuration, dimensions, gather=True)

    Creates quantities containing longitude and latitude of
    tile edges (without halo points), in degrees.

    Inputs:
    - configuration (Dict) from configure_domain()
    - dimensions (Dict)
    - gather (bool): if true, then gathers all tiles

    Outputs:
    - lon in degrees
    - lat in degrees
    """

    lon = init_quantity(
        dimensions, VariableGrid.CellCorners, VariableDims.XY, units=units["coord-deg"]
    )
    lon.data[:] = domain_configuration["metric_terms"].lon.data * 180 / np.pi

    lat = init_quantity(
        dimensions, VariableGrid.CellCorners, VariableDims.XY, units=units["coord-deg"]
    )
    lat.data[:] = domain_configuration["metric_terms"].lat.data * 180 / np.pi

    if gather:
        lon = domain_configuration["communicator"].gather(lon)
        lat = domain_configuration["communicator"].gather(lat)

    return lon, lat


def check_get_data_from_quantity(field: Union[Quantity, np.ndarray]) -> np.ndarray:
    """
    Use:
    field = check_get_data_from_quantity(field)

    If field is a quantity, gets data out as an array.
    If field is an array, it remains an array.

    Inputs:
    - field (Quantity or array)

    Outputs:
    - field: array
    """
    if isinstance(field.data, np.ndarray):
        field = field.data

    return field


def check_fill_data_to_quantity(
    field: Union[Quantity, np.ndarray], data
) -> Union[Quantity, np.ndarray]:
    """
    Use:
    field = check_fill_data_to_quantity(field, data)

    If field is a quantity, data gets stored inside field.data.
    If field is an array, it remains an array.

    """
    if isinstance(field.data, np.ndarray):
        field.data[:] = data
    else:
        field = data

    return field


def create_initial_tracer(
    lon: Union[Quantity, np.ndarray],
    lat: Union[Quantity, np.ndarray],
    tracer: Union[Quantity, np.ndarray],
    center: Tuple[float, float] = (0.0, 0.0),
) -> Union[Quantity, np.ndarray]:
    """
    Use: tracer =
            create_initial_tracer(lon, lat, tracer, target_tile)

    Calculates a gaussian-bell shaped multiplier for tracer initialization.
    It centers the bell at the longitude and latitude provided in center.

    Inputs:
    - lon and lat (in radians, and including halo points)
    - tracer: empty array to be filled with tracer
    - center: (lon, lat) in degrees

    Outputs:
    - tracer: updated quantity
    """

    lon = check_get_data_from_quantity(lon)
    lat = check_get_data_from_quantity(lat)
    tracer_input = check_get_data_from_quantity(tracer)

    r0 = RADIUS / 3.0

    p_center = [np.deg2rad(center[0]), np.deg2rad(center[1])]

    for jj in range(tracer_input.shape[1] - 1):
        for ii in range(tracer_input.shape[0] - 1):

            p_dist = [lon[ii, jj], lat[ii, jj]]
            r = great_circle_distance_lon_lat(
                p_center[0], p_dist[0], p_center[1], p_dist[1], RADIUS, np
            )

            tracer_input[ii, jj, :] = (
                0.5 * (1.0 + np.cos(np.pi * r / r0)) if r < r0 else 0.0
            )

    tracer = check_fill_data_to_quantity(tracer, tracer_input)

    return tracer


def calculate_streamfunction(
    lon_agrid: Union[Quantity, np.ndarray],
    lat_agrid: Union[Quantity, np.ndarray],
    lon: Union[Quantity, np.ndarray],
    lat: Union[Quantity, np.ndarray],
    psi: Union[Quantity, np.ndarray],
    psi_staggered: Union[Quantity, np.ndarray],
    test_case: str,
) -> Tuple[Union[Quantity, np.ndarray], Union[Quantity, np.ndarray]]:
    """
    Use: psi, psi_staggered =
            calculate_streamfunction(
                lon_agrid, lat_agrid, lon, lat,
                psi, psi_staggered, test_case
            )

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

    lat_agrid = check_get_data_from_quantity(lat_agrid)
    lon_agrid = check_get_data_from_quantity(lon_agrid)
    lat = check_get_data_from_quantity(lat)
    lon = check_get_data_from_quantity(lon)
    psi_input = check_get_data_from_quantity(psi)
    psi_staggered_input = check_get_data_from_quantity(psi_staggered)

    yA_t = np.cos(lat_agrid) * np.sin(lon_agrid)
    zA_t = np.sin(lat_agrid)
    y_t = np.cos(lat) * np.sin(lon)
    z_t = np.sin(lat)

    if test_case == "a":
        RadA = RADIUS * np.ones(lon_agrid.shape)
        Rad = RADIUS * np.ones(lon.shape)
        multiplierA = zA_t
        multiplier = z_t
    elif test_case == "b":
        RadA = RADIUS * np.cos(lat_agrid / 2)
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
        RadA = np.ones(lon_agrid.shape) * np.nan
        Rad = np.ones(lon.shape) * np.nan
        multiplierA = np.nan
        multiplier = np.nan
        print("Please choose one of the defined test cases.")
        print("This will return gibberish.")

    Ubar = (2.0 * np.pi * RADIUS) / (12.0 * 86400.0)
    streamfunction_agrid = -1 * Ubar * RadA * multiplierA
    psi_input[:, :, :] = streamfunction_agrid[:, :, np.newaxis]
    streamfunction = -1 * Ubar * Rad * multiplier
    psi_staggered_input[:, :, :] = streamfunction[:, :, np.newaxis]

    psi = check_fill_data_to_quantity(psi, psi_input)
    psi_staggered = check_fill_data_to_quantity(psi_staggered, psi_staggered_input)

    return psi, psi_staggered


def calculate_winds_from_streamfunction_grid(
    psi: Union[Quantity, np.ndarray],
    dx: Union[Quantity, np.ndarray],
    dy: Union[Quantity, np.ndarray],
    u_grid: Union[Quantity, np.ndarray],
    v_grid: Union[Quantity, np.ndarray],
    grid: GridType = GridType.AGrid,
) -> Tuple[Union[Quantity, np.ndarray], Union[Quantity, np.ndarray]]:
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

    if isinstance(u_grid.data, np.ndarray) and isinstance(v_grid.data, np.ndarray):
        if grid == GridType.AGrid:
            if not (
                u_grid.metadata.dims == ("x", "y", "z")
                and v_grid.metadata.dims == ("x", "y", "z")
            ):
                print("Incorrect wind input dimensions for A-grid.")
        elif grid == GridType.CGrid:
            if not (
                u_grid.metadata.dims == ("x", "y_interface", "z")
                and v_grid.metadata.dims == ("x_interface", "y", "z")
            ):
                print("Incorrect wind input dimensions for C-grid.")
        elif grid == GridType.DGrid:
            if not (
                u_grid.metadata.dims == ("x_interface", "y", "z")
                and v_grid.metadata.dims == ("x", "y_interface", "z")
            ):
                print("Incorrect wind input dimensions for D-grid.")
    else:
        if grid == GridType.AGrid:
            if (
                not u_grid.shape[0] == u_grid.shape[1]
                and v_grid.shape[0] == v_grid.shape[1]
            ):
                print("Incorrect wind input dimensions for A-grid.")
        elif grid == GridType.CGrid:
            if (
                not u_grid.shape[0] + 1 == u_grid.shape[1]
                and v_grid.shape[0] == v_grid.shape[1] + 1
            ):
                print("Incorrect wind input dimensions for C-grid.")
        elif grid == GridType.DGrid:
            if (
                not u_grid.shape[0] == u_grid.shape[1] + 1
                and v_grid.shape[0] + 1 == v_grid.shape[1]
            ):
                print("Incorrect wind input dimensions for D-grid.")

    psi = check_get_data_from_quantity(psi)
    dx = check_get_data_from_quantity(dx)
    dy = check_get_data_from_quantity(dy)
    u_grid_input = check_get_data_from_quantity(u_grid)
    v_grid_input = check_get_data_from_quantity(v_grid)

    if grid == GridType.AGrid:
        u_grid_input[:, 1:-1, :] = (
            -0.5 * (psi[:, 2:, :] - psi[:, :-2, :]) / dy[:, 1:-1, np.newaxis]
        )
        v_grid_input[1:-1, :, :] = (
            0.5 * (psi[2:, :, :] - psi[:-2, :, :]) / dx[1:-1, :, np.newaxis]
        )

    elif grid == GridType.CGrid:
        u_grid_input[:, :-1, :] = (
            -1 * (psi[:, 1:, :] - psi[:, :-1, :]) / dy[:, :-1, np.newaxis]
        )
        v_grid_input[:-1, :, :] = (psi[1:, :, :] - psi[:-1, :, :]) / dx[
            :-1, :, np.newaxis
        ]

    elif grid == GridType.DGrid:
        u_grid_input[:, 1:, :] = (
            -(psi[:, 1:, :] - psi[:, :-1, :]) / dy[:, 1:, np.newaxis]
        )
        v_grid_input[1:, :, :] = (
            -(psi[1:, :, :] - psi[:-1, :, :]) / dy[1:, :, np.newaxis]
        )

    u_grid = check_fill_data_to_quantity(u_grid, u_grid_input)
    v_grid = check_fill_data_to_quantity(v_grid, v_grid_input)

    return u_grid, v_grid


def create_initial_state_advection(
    metric_terms: MetricTerms,
    dimensions: Dict[str, int],
    tracer_center: Tuple[float, float],
    test_case: str,
) -> Dict[str, Quantity]:
    """
    Use: initial_state =
            create_initial_state_advection(metric_terms, dimensions, tracer_center,
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
    - dimensions (Dict) from define_dimensions()
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

    lon_agrid = metric_terms.lon_agrid
    lat_agrid = metric_terms.lat_agrid

    lon = metric_terms.lon
    lat = metric_terms.lat

    # tracer
    tracer = init_quantity(
        dimensions, VariableGrid.CellCenters, VariableDims.XYZ, units=units["tracer"]
    )
    tracer = create_initial_tracer(
        lon_agrid,
        lat_agrid,
        tracer,
        tracer_center,
    )

    # # pressure
    delp = init_quantity(
        dimensions, VariableGrid.CellCenters, VariableDims.XYZ, units=units["pressure"]
    )
    delp.data[:-1, :-1, :-1] = 10

    # # # streamfunction
    psi_agrid = init_quantity(
        dimensions,
        VariableGrid.CellCenters,
        VariableDims.XYZ,
        units=units["streamfunction"],
    )
    psi = init_quantity(
        dimensions,
        VariableGrid.CellCorners,
        VariableDims.XYZ,
        units=units["streamfunction"],
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
    dx = metric_terms.dx
    dy = metric_terms.dy

    u_cgrid = init_quantity(
        dimensions, VariableGrid.StaggeredInY, VariableDims.XYZ, units=units["wind"]
    )
    v_cgrid = init_quantity(
        dimensions, VariableGrid.StaggeredInX, VariableDims.XYZ, units=units["wind"]
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
    stencil_configuration: Dict[str, Any],
    initial_state: Dict[str, Quantity],
    dimensions: Dict[str, int],
    timestep: float,
    density: float = 1.0,
) -> Dict[str, Quantity]:
    """
    Use: fluxPrep =
            run_finite_volume_fluxprep(configuration, initial_state,
            dimensions, timestep)

    Initializes and runs the FiniteVolumeFluxPrep class to get
    initial states for mass flux, contravariant winds, area flux.

    Inputs:
    - stencil_configuration (Dict) from configure_stencil()
    - initial_state (Dict) from create_initial_state()
    - dimensions (Dict)
    - timestep (float) for advection

    Outputs:
    - flux_prep (Dict):
        - crx and cry
        - mfxd and mfyd
        - uc_contra, vc_contra
        - x_area_flux, y_area_flux
    """

    crx = init_quantity(
        dimensions, VariableGrid.StaggeredInX, VariableDims.XYZ, units=units["courant"]
    )
    cry = init_quantity(
        dimensions, VariableGrid.StaggeredInY, VariableDims.XYZ, units=units["courant"]
    )

    x_area_flux = init_quantity(
        dimensions, VariableGrid.StaggeredInX, VariableDims.XYZ, units=units["area"]
    )
    y_area_flux = init_quantity(
        dimensions, VariableGrid.StaggeredInY, VariableDims.XYZ, units=units["area"]
    )

    uc_contra = init_quantity(
        dimensions, VariableGrid.StaggeredInX, VariableDims.XYZ, units=units["area"]
    )
    vc_contra = init_quantity(
        dimensions, VariableGrid.StaggeredInY, VariableDims.XYZ, units=units["area"]
    )

    # intialize and run
    fvf_prep = FiniteVolumeFluxPrep(
        stencil_configuration["stencil_factory"], stencil_configuration["grid_data"]
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
    )  # this will modify empty quantities, but not change uc, vc

    mfxd = init_quantity(
        dimensions, VariableGrid.StaggeredInX, VariableDims.XYZ, units=units["area"]
    )
    mfxd.data[:] = x_area_flux.data[:] * initial_state["delp"].data[:] * density
    mfyd = init_quantity(
        dimensions, VariableGrid.StaggeredInY, VariableDims.XYZ, units=units["area"]
    )
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
    stencil_configuration: Dict[str, Any], tracers: Dict[str, Quantity]
) -> TracerAdvection:
    """
    Use: tracer_advection =
            build_tracer_advection(stencil_configuration, tracers)


    Initializes FiniteVolumeTransport and TracerAdvection classes.

    Inputs:
    - stencil_configuration (Dict) from configure_stencil()
    - tracers (Dict) from initialState created by create_initial_state()

    Outputs:
    - tracer_advection - an instance of TracerAdvection class
    """
    fvt_dict = {"grid_type": 0, "hord": 6}

    fvtp_2d = FiniteVolumeTransport(
        stencil_configuration["stencil_factory"],
        stencil_configuration["quantity_factory"],
        stencil_configuration["grid_data"],
        stencil_configuration["damping_coefficients"],
        fvt_dict["grid_type"],
        fvt_dict["hord"],
    )

    tracer_advection = TracerAdvection(
        stencil_configuration["stencil_factory"],
        stencil_configuration["quantity_factory"],
        fvtp_2d,
        stencil_configuration["grid_data"],
        stencil_configuration["communicator"],
        tracers,
    )

    return tracer_advection


def prepare_everything_for_advection(
    stencil_configuration: Dict[str, Any],
    initial_state: Dict[str, Quantity],
    dimensions: Dict[str, int],
    timestep: float,
) -> Tuple[Dict[str, Any], TracerAdvection]:
    """
    Use: tracer_advection_data, tracer_advection =
        prepare_everything_for_advection(stencil_configuration, initial_state,
        dimensions, timestep)

    Calls run_finite_volume_fluxprep() and build_tracer_advection().

    Inputs:
    - stencil_configuration from configure_stencil()
    - initialState from create_initial_state()
    - dimensions
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
        stencil_configuration,
        initial_state,
        dimensions,
        timestep,
    )

    tracer_advection = build_tracer_advection(stencil_configuration, tracers)

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
    )

    tracer_advection_data["delp"] = cp.deepcopy(tracer_advection_data_initial["delp"])
    tracer_advection_data["mfxd"] = cp.deepcopy(tracer_advection_data_initial["mfxd"])
    tracer_advection_data["mfyd"] = cp.deepcopy(tracer_advection_data_initial["mfyd"])
    tracer_advection_data["crx"] = cp.deepcopy(tracer_advection_data_initial["crx"])
    tracer_advection_data["cry"] = cp.deepcopy(tracer_advection_data_initial["cry"])

    return tracer_advection_data


def plot_grid(
    lon: Quantity,
    lat: Quantity,
    dimensions: Dict[str, int],
    fOut: str = "grid_map.png",
    show: bool = False,
) -> None:
    """
    Use: plot_grid(lon, lat, dimensions,
            fOut="grid_map.png", show=False)

    Creates a Robinson projection and plots grid edges.
    Note -- this is basically useless for more than
    50 points as the cells are too small.

    Inputs:
    - lon, lat (Quantity)
    - dimensions (Dict)
    - fOut (str): file name to save to
    - show (bool): whether to show image in notebook

    Outputs: saved figure
    """
    lon = check_get_data_from_quantity(lon)
    lat = check_get_data_from_quantity(lat)

    field = np.zeros(lon.shape)[:, :-1, :-1]

    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    ax.set_facecolor(".4")

    pcolormesh_cube(
        lat,
        lon,
        field,
        cmap="bwr",
        vmin=-1,
        vmax=1,
        edgecolor="k",
        linewidth=0.1,
    )

    nx = dimensions["nx"]
    ny = dimensions["ny"]
    ax.set_title(f"Cubed-sphere mesh with {nx} x {ny} cells per tile (c{nx})")

    plt.savefig(fOut, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close("all")


def plot_projection_field(
    lon: Union[Quantity, np.ndarray],
    lat: Union[Quantity, np.ndarray],
    field: Union[Quantity, np.ndarray],
    plot_dict: Dict[str, Any],
    f_out: str,
    show: bool = False,
    unstagger: str = "first",
    level: int = 0,
) -> None:
    """
    Use: plot_projection_field(lon, lat,
            field, plot_dict, fOut,
            show=False, unstagger="first", level=0)

    Creates a Robinson projection and plots provided field.
    The field must have been gathered from subprocesses first.

    Inputs:
    - lon, lat (Quantity or array)
    - field (Quantity or array) - if multiple layers, gets subset
    - plot_dict (Dict) - vmin, vmax, etc.
    - f_out (str): file name to save to - if blank, not saved.
    - show (bool): whether to show image in notebook
    - unstagger: if field is staggered, it unstaggers it
    - level: if multilple levels, which gets subset

    Outputs: saved figure
    """

    lon = check_get_data_from_quantity(lon)
    lat = check_get_data_from_quantity(lat)
    field = check_get_data_from_quantity(field)

    if len(field.shape) == 4:
        field = np.squeeze(field[:, :, :, level])

    field_plot = unstagger_coordinate(field, unstagger)

    if "vmin" not in plot_dict:
        plot_dict["vmin"] = 0
    if "vmax" not in plot_dict:
        plot_dict["vmax"] = 1
    if "units" not in plot_dict:
        plot_dict["units"] = "forgot to add units"
    if "title" not in plot_dict:
        plot_dict["title"] = "forgot to add title"
    if "cmap" not in plot_dict:
        plot_dict["cmap"] = "viridis"

    fig = plt.figure(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    ax.set_facecolor(".4")

    f1 = pcolormesh_cube(
        lat,
        lon,
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
    lon: Union[Quantity, np.ndarray],
    lat: Union[Quantity, np.ndarray],
    tracer_gather: Union[List[Quantity], List[np.ndarray]],
    plot_dict_tracer: Dict[str, Any],
    figure_everyNsteps: int,
    timestep: float,
    frames: Union[str, int] = "all",
) -> None:
    """
    Use: plot_tracer_animation(lon, lat,
            tracer_gather, plot_dict_tracer,
            figure_everyNsteps, timestep, frames="all")

    Plots an interactive animation inside Jupyter notebook.

    Inputs:
    - lon, lat (Quantity or array)
    - tracer_gather (List of Quantities) of tracer states
    - plot_dict_tracer (Dict) of plotting settings
    - figure_everyNsteps (int) from initial setup
    - timestep (float) for advection
    - frames ("all" or int) - how many frames to plot

    Outputs: animation
    """

    lon = check_get_data_from_quantity(lon)
    lat = check_get_data_from_quantity(lat)

    tracer_global = []
    for step in range(len(tracer_gather)):
        tracer_global.append(check_get_data_from_quantity(tracer_gather[step]))

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
            lat,
            lon,
            tracer_stack[step][:, :, :],
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


def unstagger_coordinate(
    field: Union[Quantity, np.ndarray], mode: str = "mean"
) -> np.ndarray:
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

    field_input = check_get_data_from_quantity(field)

    fs = field_input.shape

    if len(fs) == 2:
        field = field[np.newaxis]
        zDim, dim1, dim2 = field.shape
    elif len(fs) == 3:
        zDim, dim1, dim2 = field.shape
    elif len(fs) == 4:
        zDim, dim1, dim2, dim3 = field.shape

    if mode == "mean":
        if dim1 > dim2:
            field_unstagger = 0.5 * (field_input[:, 1:, :] + field_input[:, :-1, :])
        elif dim2 > dim1:
            field_unstagger = 0.5 * (field_input[:, :, 1:] + field_input[:, :, :-1])
        elif dim1 == dim2:
            field_unstagger = field_input

    elif mode == "first":
        if dim1 > dim2:
            field_unstagger = field[:, :-1, :]
        elif dim2 > dim1:
            field_unstagger = field[:, :, :-1]
        elif dim1 == dim2:
            field_unstagger = field_input

    elif mode == "last":
        if dim1 > dim2:
            field_unstagger = field[:, 1:, :]
        elif dim2 > dim1:
            field_unstagger = field[:, :, 1:]
        elif dim1 == dim2:
            field_unstagger = field_input

    if len(fs) == 2:
        field_unstagger = field_unstagger[0]

    field = check_fill_data_to_quantity(field, field_unstagger)

    return field


def remap_winds_to_meteorological(
    u_input: Union[Quantity, np.ndarray],
    v_input: Union[Quantity, np.ndarray],
    nTiles: int = 6,
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

    u_input = check_get_data_from_quantity(u_input)
    v_input = check_get_data_from_quantity(v_input)

    zonal = np.zeros(u_input.shape)
    meridional = np.zeros(v_input.shape)

    for tile in range(nTiles):
        if tile in [0, 1, 5]:  # first two tiles are normal
            zonal[tile] = u_input[tile]
            meridional[tile] = v_input[tile]

        if tile == 2:
            zonal[tile] = -v_input[tile]
            meridional[tile] = u_input[tile]

        if tile in [3, 4]:
            zonal[tile] = v_input[tile]
            meridional[tile] = -u_input[tile]

    zonal = check_fill_data_to_quantity(zonal)
    meridional = check_fill_data_to_quantity(meridional)

    return zonal, meridional
