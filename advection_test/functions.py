

#from netCDF4 import Dataset
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.stencils.fxadv import FiniteVolumeFluxPrep
from fv3core.stencils.tracer_2d_1l import TracerAdvection
from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.stencil import GridIndexing, StencilConfig, StencilFactory
from pace.util import CubedSphereCommunicator, CubedSpherePartitioner, Quantity, QuantityFactory, SubtileGridSizer, TilePartitioner
from pace.util.constants import RADIUS
from pace.util.grid.gnomonic import great_circle_distance_lon_lat
from pace.util.grid import DampingCoefficients, GridData, MetricTerms
import copy as cp

import matplotlib.pyplot as plt
import numpy as np

#import ipyparallel as ipp
#import os

def get_lonLat_agrid(grid_data, dimensions, units, origins, backend):
    """
    Use: lon, lat = get_lonLat_agrid(grid_data, dimensions, units, origins, backed, final_unit='deg')

    Inputs:
    - grid_data configuration
    - dimensions: Dict{'nx1', 'ny1'}
    - units: Dict{'coord}
    - origins: Dict{'compute_2d'}
    - backend: 'numpy'

    Outputs:
    - lon and lat on A-grid in degrees
    """

    lon =  Quantity(grid_data.lon_agrid.data * 180 / np.pi, ('x_interface', 'y_interface'), units['coord'], origins['compute_2d'], (dimensions['nx1'], dimensions['ny1']), backend)
    lat =  Quantity(grid_data.lat_agrid.data * 180 / np.pi, ('x_interface', 'y_interface'), units['coord'], origins['compute_2d'], (dimensions['nx1'], dimensions['ny1']), backend)

    return lon, lat

def define_dimensionsUnitsOrigins(nx, ny, nz, nhalo, mpi_comm):
    """
    Use: dimensions, units, origins, mpi_rank = define_dimensionsUnitsOrigins(nx, ny, nz, nhalo, mpi_comm)

    Outputs dictionaries for basic configuration parameters.

    Inputs:
    - nx, ny, nz: number of points in x, y, z dimensions for each tile (add 1 like in Fotran)
    - nhalo: number of halo points
    - mpi_comm: mpi communicator

    Outputs:
    - dimensions: Dict{'nx', 'ny', 'nz', 'nx1', 'ny1', 'nhalo', 'nxhalo', nyhalo', 'tile'}
    - units: Dict{'dist', 'coord', 'mass', 'psi', 'wind', 'courant', 'areaflux', 'pressure'}
    - origins: Dict{'halo', 'compute_2d', 'compute_3d'}
    - mpi_rank: rank for each of the subprocesses
    """

    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()

    dimensions = {
                  'nx': nx, 
                  'ny': ny, 
                  'nz': nz, 
                  'nx1': nx + 1, 
                  'ny1': ny + 1, 
                  'nhalo': nhalo, 
                  'nxhalo': nx + 2*nhalo, 
                  'nyhalo': ny + 2*nhalo,
                  'tile': mpi_size
                  }
    
    units = {
             'areaflux': 'm2', 
             'coord': 'degrees', 
             'courant': '', 
             'dist': 'm', 
             'mass': 'kg', 
             'pressure': 'Pa',
             'psi': 'kg/m/s', 
             'wind': 'm/s'
             }
    
    origins = {
               'halo': (0, 0), 
               'compute_2d': (dimensions['nhalo'], dimensions['nhalo']), 
               'compute_3d': (dimensions['nhalo'], dimensions['nhalo'], 0)
               }

    return dimensions, units, origins, mpi_rank 


def configure_domain(layout, mpi_comm, dimensions, backend='numpy'):
    """
    Use: configuration = configure_domain(layout, mpi_comm, dimensions, backend='numpy')

    Inputs:
    - layout:
    - mpi_comm:
    - dimensions: Dict{'nx', 'ny', 'nx1', 'ny1', 'nz', 'nhalo'}
    - backend

    Outputs: configuration dictionary that includes:
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


    sizer = SubtileGridSizer.from_tile_params(nx_tile=dimensions['nx']-1, ny_tile=dimensions['ny']-1, nz=dimensions['nz']-1, n_halo=dimensions['nhalo'], 
                                              extra_dim_lengths={}, layout=layout, tile_partitioner=partitioner.tile, tile_rank=communicator.tile.rank)

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)

    metric_terms = MetricTerms(quantity_factory=quantity_factory, communicator=communicator)
    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
    grid_data = GridData.new_from_metric_terms(metric_terms)

    dace_config = DaceConfig(communicator=None, backend=backend, orchestration=DaCeOrchestration.Python)
    stencil_config = StencilConfig(backend=backend, rebuild=False, validate_args=True, dace_config=dace_config)
    grid_indexing = GridIndexing.from_sizer_and_communicator(sizer=sizer, cube=communicator)

    ### set the domain so there is only one level in the vertical -- forced 
    domain = (grid_indexing.domain)
    domain_new = list(domain)
    domain_new[2] = 1
    domain_new = tuple(domain_new)

    grid_indexing.domain = domain_new

    stencil_factory = StencilFactory(config=stencil_config, grid_indexing=grid_indexing)

    configuration = {'partitioner': partitioner, 'communicator': communicator, 
                   'sizer': sizer, 'quantity_factory': quantity_factory, 
                   'metric_terms': metric_terms, 'damping_coefficients': damping_coefficients, 'grid_data': grid_data, 
                   'dace_config': dace_config, 'stencil_config': stencil_config, 'grid_indexing': grid_indexing, 'stencil_factory': stencil_factory}

    return configuration


def store_coordinates(fOut, dimensions, variables):
    """
    Use: store_coordinates(fOut, dimensions, variables)

    Creates and writes to a coordinate file.

    Inputs:
    - fOut: output netcdf file
    - dimensions: Dict{'tile', 'nxhalo', 'nyhalo'}
    - variables: Dict{'dx', 'dy', 'dxa', 'dya', 'dxc', 'dyc', 'lon', 'lat', lona', 'lata'}

    Outputs: none
    """

    if os.path.isfile(fOut):
        os.remove(fOut)

    data = Dataset(fOut, 'w')

    data.createDimension('tile', dimensions['tile'])
    data.createDimension('nxhalo', dimensions['nxhalo'])
    data.createDimension('nyhalo', dimensions['nyhalo'])

    v0 = data.createVariable('dx', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v0[:] = variables['dx']
    v0.units = 'm'
    v0 = data.createVariable('dy', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v0[:] = variables['dy']
    v0.units = 'm'

    v0 = data.createVariable('dxa', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v0[:] = variables['dxa']
    v0.units = 'm'
    v0 = data.createVariable('dya', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v0[:] = variables['dya']
    v0.units = 'm'

    v0 = data.createVariable('dxc', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v0[:] = variables['dxc']
    v0.units = 'm'
    v0 = data.createVariable('dyc', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v0[:] = variables['dyc']
    v0.units = 'm'

    v1 = data.createVariable('lon', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v1[:] = variables['lon']
    v1.units = 'degrees_east'
    v1 = data.createVariable('lat', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v1[:] = variables['lat']
    v1.units = 'degrees_north'

    v1 = data.createVariable('lona', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v1[:] = variables['lona']
    v1.units = 'degrees_east'
    v1 = data.createVariable('lata', 'f8', ('tile', 'nxhalo', 'nyhalo'))
    v1[:] = variables['lata']
    v1.units = 'degrees_north'

    data.close()

    return


def create_gaussianMultiplier(lon, lat, dimensions, mpi_rank, center_tile=0):
    """
    Use: gaussian_multiplier = gaussian_multiplier(lon, lat, dimensions, mpi_rank, center_tile=0)

    Creates a 2-D Gaussian bell shape on the desired tile/rank in the domain.

    Inputs:
    - lon, lat: longitude and latitude of centerpoints (A-grid) (in radians)
    - dimensions: Dict{'nxhalo', 'nyhalo', 'tile'}
    - mpi_rank: rank of the process
    - center_tile = 0: the tile on which to center the blob

    Outputs:
    - gaussian_multiplier: blob centered at the middle of center_tile with gaussian dropoff
    """

    r0 = RADIUS / 3.
    p1x, p1y = int(dimensions['nxhalo']/2), int(dimensions['nyhalo']/2) # center gaussian on middle of tile
    gaussian_multiplier = np.zeros((dimensions['nxhalo'], dimensions['nyhalo']))

    if mpi_rank == center_tile:
        p_center = [lon[p1x, p1y], lat[p1x, p1y]]
        print('Centering gaussian on lon=%.2f, lat=%.2f' % (np.rad2deg(p_center[0]), np.rad2deg(p_center[1])))

        for jj in range(dimensions['nyhalo']):
            for ii in range(dimensions['nxhalo']):

                p_dist = [lon[ii, jj], lat[ii, jj]]
                r = great_circle_distance_lon_lat(p_center[0], p_dist[0], p_center[1], p_dist[1], RADIUS, np) 

                gaussian_multiplier[ii, jj] = 0.5 * (1.0 + np.cos(np.pi * r / r0)) if r < r0 else 0.0         

    return gaussian_multiplier


def calculate_streamfunction_testCase1(lon, lat, dimensions):
    """
    Use: psi, psi_staggered = calculate_streamfunction_testCase1(lon, lat, dimensions)

    Calculates streamfunction for testCase1 in fortran. Runs on each rank independently.

    Inputs:
    - lon: longitude of center points (in radians)
    - lat: latitude of center points (in radians)
    - dimensions: Dict{'nxhalo', 'nyhalo'}

    Outputs:
    - psi: streamfunction on tile centers (with halo points)
    - psi_staggered: streamfunction on tile corners (with halo points)
    """

    Ubar = (2.0 * np.pi * RADIUS) / (12. * 86400.0) # 38.6 
    alpha = 0

    psi = np.ones((dimensions['nxhalo'], dimensions['nyhalo'])) * 1.e25
    psi_staggered = np.ones((dimensions['nxhalo'], dimensions['nyhalo'])) * 1.e25


    for jj in range(dimensions['nyhalo']):
        for ii in range(dimensions['nxhalo']):
            psi[ii, jj] = (-1. * Ubar * RADIUS * (np.sin(lat[ii, jj]) * np.cos(alpha) - np.cos(lon[ii, jj]) * np.cos(lat[ii, jj]) * np.sin(alpha)))

    for jj in range(dimensions['nyhalo']):
            for ii in range(dimensions['nxhalo']):
                psi_staggered[ii, jj] = (-1. * Ubar * RADIUS * (np.sin(lat[ii, jj]) * np.cos(alpha) - np.cos(lon[ii, jj]) * np.cos(lat[ii, jj]) * np.sin(alpha)))

    return psi, psi_staggered


def calculate_windsFromStreamfunction_grid(psi, dx, dy, dimensions, grid='A'):
    """
    Use: u_grid, v_grid = calculate_windsFromStreamfunction_Agrid(psi, dx, dy, dimensions, grid='A')

    Returns winds on a chosen grid based on streamfunction and grid spacing.

    Inputs:
    - psi: streamfunction
    - dx, dy: distance between points
    - dimensions: Dict{'nxhalo', 'nyhalo', 'nx', 'ny}

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
    u_grid = np.zeros((dimensions['nxhalo'], dimensions['nyhalo']))
    v_grid = np.zeros((dimensions['nxhalo'], dimensions['nyhalo']))

    if grid == 'A':
        for jj in range(dimensions['nhalo']-1, dimensions['ny']+dimensions['nhalo']+1):
            for ii in range(dimensions['nhalo']-1, dimensions['nx']+dimensions['nhalo']+1):
                psi1 = 0.5 * (psi.data[ii, jj] + psi.data[ii, jj-1])
                psi2 = 0.5 * (psi.data[ii, jj] + psi.data[ii, jj+1])
                dist = dy.data[ii, jj]
                u_grid[ii, jj] = 0 if dist == 0 else -1.0 * (psi2 - psi1) / dist

                psi1 = 0.5 * (psi.data[ii, jj] + psi.data[ii-1, jj])
                psi2 = 0.5 * (psi.data[ii, jj] + psi.data[ii+1, jj])
                dist = dx.data[ii, jj]
                v_grid[ii, jj] = 0 if dist == 0 else (psi2 - psi1) / dist
    
    if grid == 'C':
        for jj in range(dimensions['nhalo']-1, dimensions['ny']+dimensions['nhalo']+1):
            for ii in range(dimensions['nhalo']-1, dimensions['nx']+dimensions['nhalo']+1):
                dist = dx.data[ii, jj]
                v_grid[ii, jj] = 0 if dist == 0 else (psi.data[ii+1, jj] - psi.data[ii, jj]) / dist

                dist = dy.data[ii, jj]
                u_grid[ii, jj] = 0 if dist == 0 else -1.0 * (psi.data[ii, jj+1] - psi.data[ii, jj]) / dist
    
    if grid == 'D':
        for jj in range(dimensions['nhalo']-1, dimensions['ny']+dimensions['nhalo']+1):
            for ii in range(dimensions['nhalo']-1, dimensions['nx']+dimensions['nhalo']+1):
                dist = dx.data[ii, jj]
                v_grid[ii, jj] = 0 if dist == 0 else (psi.data[ii, jj] - psi.data[ii-1, jj]) / dist

                dist = dy.data[ii, jj]
                u_grid[ii, jj] = 0 if dist == 0 else -1.0 * (psi.data[ii, jj] - psi.data[ii, jj-1]) / dist     
        
    return u_grid, v_grid


def create_initialState_testCase1(grid_data, dimensions, units, origins, backend, smoke_dict, pressure_dict):
    """
    Use: create_initialState(grid_data, dimensions, units, origins, backend, smoke_dict, pressure_dict)

    Creates inital state from the fortran test_case 1 streamfunction configuration - pressure, gaussian smoke distribution, u and v winds on C-grid.

    Inputs:
    - grid_data: configuration['grid_data']
    - dimensions: Dict{'nx', 'ny', 'nx1', 'ny1', 'nxhalo', 'nyhalo', 'tile'}
    - units: Dict{'coord', 'dist', 'mass', 'psi', 'wind'}
    - origins: Dict{'halo', 'compute_2d'}
    - backend: 'numpy'
    - smoke_dict: Dict{'center_tile', 'rank', 'smoke_base'}
    - pressure_dict: Dict{'pressure_base'}

    Outputs:
    - initialState: Dict{'delp', 'smoke', 'uc', 'vc'} - 3D with a single layer in the vertical
    """
    lonA_halo =  Quantity(grid_data.lon_agrid.data, ('x_halo', 'y_halo'), units['coord'], origins['halo'], (dimensions['nxhalo'], dimensions['nyhalo']), backend)
    latA_halo =  Quantity(grid_data.lat_agrid.data, ('x_halo', 'y_halo'), units['coord'], origins['halo'], (dimensions['nxhalo'], dimensions['nyhalo']), backend)

    # SMOKE
    gaussian_multiplier = create_gaussianMultiplier(lonA_halo.data, latA_halo.data, dimensions, center_tile=smoke_dict['center_tile'], mpi_rank=smoke_dict['rank'])
    smoke = Quantity(gaussian_multiplier * smoke_dict['smoke_base'], ('x', 'y'), units['mass'], origins['compute_2d'], (dimensions['nx'], dimensions['ny']), backend)
    
    # PRESSURE
    delp = Quantity(np.ones(smoke.data.shape) * pressure_dict['pressure_base'], ('x', 'y'), units['pressure'], origins['compute_2d'], (dimensions['nx'], dimensions['ny']), backend)

    # STREAMFUNCTION
    _, psi_staggered = calculate_streamfunction_testCase1(lonA_halo.data, latA_halo.data, dimensions)
    psi_staggered_halo = Quantity(psi_staggered, ('x_halo', 'y_halo'), units['psi'], origins['halo'], (dimensions['nxhalo'], dimensions['nyhalo']), backend)

    # WINDS
    dx_halo =  Quantity(grid_data.dx.data, ('x_halo', 'y_halo'), units['dist'], origins['halo'], (dimensions['nxhalo'], dimensions['nyhalo']), backend)
    dy_halo =  Quantity(grid_data.dy.data, ('x_halo', 'y_halo'), units['dist'], origins['halo'], (dimensions['nxhalo'], dimensions['nyhalo']), backend)
    uC, vC = calculate_windsFromStreamfunction_grid(psi_staggered_halo, dx_halo, dy_halo, dimensions, grid='C')
    uC = Quantity(uC, ('x', 'y_interface'), units['wind'], origins['compute_2d'], (dimensions['nx'], dimensions['ny1']), backend)
    vC = Quantity(vC, ('x_interface', 'y'), units['wind'], origins['compute_2d'], (dimensions['nx1'], dimensions['ny']), backend)

    # EXTEND INITIAL CONDITIONS INTO ONE VERTICAL LAYER
    dimensions['nz'] = 1
    empty = np.zeros((dimensions['nxhalo'], dimensions['nyhalo'], dimensions['nz']+1))

    smoke_3d = np.copy(empty)
    uC_3d = np.copy(empty)
    vC_3d = np.copy(empty)
    delp_3d = np.copy(empty)

    smoke_3d[:, :, 0] = smoke.data
    uC_3d[:, :, 0] = uC.data
    vC_3d[:, :, 0] = vC.data
    delp_3d[:, :, 0] = delp.data

    smoke = Quantity(smoke_3d, ('x', 'y', 'z'), units['mass'], origins['compute_3d'], (dimensions['nx'], dimensions['ny'], dimensions['nz']), backend)
    uC = Quantity(uC_3d, ('x', 'y_interface', 'z'), units['wind'], origins['compute_3d'], (dimensions['nx'], dimensions['ny1'], dimensions['nz']), backend)
    vC = Quantity(vC_3d, ('x_interface', 'y', 'z'), units['wind'], origins['compute_3d'], (dimensions['nx1'], dimensions['ny'], dimensions['nz']), backend)
    delp = Quantity(delp_3d, ('x', 'y', 'z'), units['wind'], origins['compute_3d'], (dimensions['nx'], dimensions['ny'], dimensions['nz']), backend)

    initialState = {
                    'delp': delp, 
                    'smoke': smoke, 
                    'uC': uC,
                    'vC': vC,
                    }

    return initialState


def run_finiteVolumeFluxPrep(configuration, uc, vc, delp, density, dimensions, units, origins, backend, dt_acoustic=300):
    """
    Use: fluxPrep = run_finiteVolumeFluxPrep(configuration, uc, vc, dimensions, units, origins, backend, data, dt_acoustic=300)

    Initializes FiniteVolumeFluxPrep class and fills in variables from initial wind data.

    Inputs:
    - configuration: Dict{'stencil_factory', 'grid_data'}
    - uc, vc: x- and y- winds on C-grid
    - delp: pressure thickness of layer
    - density: 1 (kg/m3 for air)
    - dimensions: Dict{'nx', 'ny', 'nz', 'nx1', 'ny1', 'nxhalo', 'nyhalo'}
    - units: Dict{'areaflux', 'courant', 'mass', 'wind'}
    - origins: Dict{'compute_3d'}
    - backend: 'numpy'
    - dt_acoustic: acoustic time step (default = 300 seconds)

    Outputs:
    - fluxPrep: Dict{'crx', 'cry', 'mfxd', 'mfyd', 'ucv', 'vcv', 'xaf', 'yaf'}
        - crx, cry are Courant numbers
        - ucv, vcv are the contravariant winds
        - xaf, yaf are fluxes of area (dx*dy but incorporating wind speed)
        - mfxd, mfyd are mass (volume) fluxes incorporating wind speed
    """
    
    # CREATE EMPTY QUANTITIES TO BE FILLED
    empty = np.zeros((dimensions['nxhalo'], dimensions['nyhalo'], dimensions['nz']+1))

    crx = Quantity(empty, ('x_interface', 'y', 'z'), units['courant'], origins['compute_3d'], (dimensions['nx1'], dimensions['ny'], dimensions['nz']), backend)
    cry = Quantity(empty, ('x', 'y_interface', 'z'), units['courant'], origins['compute_3d'], (dimensions['nx'], dimensions['ny1'], dimensions['nz']), backend)
    xaf = Quantity(empty, ('x_interface', 'y', 'z'), units['areaflux'], origins['compute_3d'], (dimensions['nx1'], dimensions['ny'], dimensions['nz']), backend)
    yaf = Quantity(empty, ('x', 'y_interface', 'z'), units['areaflux'], origins['compute_3d'], (dimensions['nx'], dimensions['ny1'], dimensions['nz']), backend)
    ucv = Quantity(empty, ('x_interface', 'y', 'z'), units['wind'], origins['compute_3d'], (dimensions['nx1'], dimensions['ny'], dimensions['nz']), backend)
    vcv = Quantity(empty, ('x', 'y_interface', 'z'), units['wind'], origins['compute_3d'], (dimensions['nx'], dimensions['ny1'], dimensions['nz']), backend)

    # INITIALIZE AND RUN
    fvf_prep = FiniteVolumeFluxPrep(configuration['stencil_factory'], configuration['grid_data'])

    fvf_prep(uc, vc, crx, cry, xaf, yaf, ucv, vcv, dt_acoustic) # THIS WILL MODIFY CREATED QUANTITIES, but not change uc, vc

    mfxd = Quantity(empty, ('x_interface', 'y', 'z'), units['mass'], origins['compute_3d'], (dimensions['nx1'], dimensions['ny'], dimensions['nz']), backend)
    mfyd = Quantity(empty, ('x', 'y_interface', 'z'), units['mass'], origins['compute_3d'], (dimensions['nx'], dimensions['ny1'], dimensions['nz']), backend)

    mfxd.data[:] = xaf.data[:] * delp.data[:] * density
    mfyd.data[:] = yaf.data[:] * delp.data[:] * density

    fluxPrep = {
                'crx': crx,
                'cry': cry,
                'mfxd': mfxd,
                'mfyd': mfyd,
                'ucv': ucv,
                'vcv': vcv,
                'xaf': xaf,
                'yaf': yaf,
                }

    return fluxPrep


def build_tracerAdvection(configuration, fvt_dict, tracers):
    """
    Use: tracAdv = build_tracerAdvection(configuration, fvt_dict, tracers)

    Initializes the tracer advection class from FiniteVolumeTransport and TracerAdvection.

    Inputs:
    - configuration: Dict{'stencil_factory', 'grid_data', 'damping_coefficients'}
    - fvt_dict: Dict{'grid_type', 'hord'}
    - tracers: Dict{'smoke}

    Outputs:
    - tracAdv: class instance that performs tracer advection
    """

    fvtp_2d = FiniteVolumeTransport(configuration['stencil_factory'], configuration['grid_data'], configuration['damping_coefficients'], fvt_dict['grid_type'], fvt_dict['hord'])

    tracAdv = TracerAdvection(configuration['stencil_factory'], fvtp_2d, configuration['grid_data'], configuration['communicator'], tracers)

    return tracAdv


def prepare_everythingForAdvection(configuration, uc, vc, delp, tracers, density, dimensions, units, origins, backend, fvt_dict, dt_acoustic=300):
    """
    Use: = prepare_everythingForAdvection(configuration, uc, vc, delp, tracers, density, dimensions, units, origins, backend, fvt_dict, dt_acoustic=300)

    Inputs:
    - configuration: Dict{'stencil_factory', 'grid_data', 'damping_coefficients'}
    - uc, vc: x- and y- winds on C-grid
    - delp: pressure thickness of layer
    - tracers: Dict{'smoke}
    - density: (kg/m3 for air)
    - dimensions: Dict{'nx', 'ny', 'nz', 'nx1', 'ny1', 'nxhalo', 'nyhalo'}
    - units: Dict{'areaflux', 'courant', 'mass', 'wind'}
    - origins: Dict{'compute_3d'}
    - backend: 'numpy'
    - fvt_dict: Dict{'grid_type', 'hord'}
    - dt_acoustic: acoustic time step (default = 300 seconds)

    Outputs:
    - fluxPrep: Dict{'crx', 'cry', 'mfxd', 'mfyd', 'ucv', 'vcv', 'xaf', 'yaf'}
        - crx, cry are Courant numbers
        - ucv, vcv are the contravariant winds
        - xaf, yaf are fluxes of area (dx*dy but incorporating wind speed)
        - mfxd, mfyd are mass (volume) fluxes incorporating wind speed
    - tracAdv: class instance that performs tracer advection
    """

    fluxPrep = run_finiteVolumeFluxPrep(configuration, uc, vc, delp, density, dimensions, units, origins, backend, dt_acoustic)

    tracAdv = build_tracerAdvection(configuration, fvt_dict, tracers)

    tracAdv_data = {
                    'tracers': tracers,
                    'delp': delp,
                    'mfxd': fluxPrep['mfxd'],
                    'mfyd': fluxPrep['mfyd'],
                    'crx': fluxPrep['crx'],
                    'cry': fluxPrep['cry']
    }

    return tracAdv_data, tracAdv


def run_advectionStepWithReset(tracAdv_dataInit, tracAdv_data, tracAdv, dt, mpi_rank=None):
    """
    Use: = run_advectionStepWithReset()

    Inputs:
    - tracAdv_dataInit: Dict{'tracers', 'delp', 'mfxd', 'mfyd', 'crx', 'cry'} of values to be reset to after each advection step
    - tracAdv_data: Dict{'tracers', 'delp', 'mfxd', 'mfyd', 'crx', 'cry'}
    - tracAdv: class instance of TracerAdvection
    - dt: time step in seconds
    - mpi_rank: if 0, prints differences from timestep, initial condition

    Outputs:
    - tracAdv_data: updated fields (of only tracer)
    """

    tmp = cp.deepcopy(tracAdv_data['tracers']) # pre-advection tracer state

    tracAdv(tracAdv_data['tracers'], tracAdv_data['delp'], tracAdv_data['mfxd'], tracAdv_data['mfyd'], tracAdv_data['crx'], tracAdv_data['cry'], dt)

    tracAdv_data['delp'] = cp.deepcopy(tracAdv_dataInit['delp'])
    tracAdv_data['mfxd'] = cp.deepcopy(tracAdv_dataInit['mfxd'])
    tracAdv_data['mfyd'] = cp.deepcopy(tracAdv_dataInit['mfyd'])
    tracAdv_data['crx'] = cp.deepcopy(tracAdv_dataInit['crx'])
    tracAdv_data['cry'] = cp.deepcopy(tracAdv_dataInit['cry'])


    if mpi_rank == 0:
        diff_timestep = tracAdv_data['tracers']['smoke'].data - tmp['smoke'].data
        diff_fromInit = tracAdv_data['tracers']['smoke'].data - tracAdv_dataInit['tracers']['smoke'].data
        print('timestep diff min = %.2e; max = %.2e; from init diff min = %.2e; max = %.2e' % (np.nanmin(diff_timestep), np.nanmax(diff_timestep), np.nanmin(diff_fromInit), np.nanmax(diff_fromInit)))

    return tracAdv_data


def write_initialCondition_toFile(fOut, variables, dimensions, units):
    """

    Inputs:
    - fOut: output netcdf file
    - variables: Dict{'ua', 'va', 'psi', 'qvapor'}
    - dimensions: Dict{'tile', 'nx', 'ny', 'nx1', 'ny1'}
    - configuration: output of configure_domain function
    - units: Dict{'coord', 'wind', 'psi', 'qvapor'}
    - origins: Dict{'compute_2d'}
    - backend = 'numpy'
    """
    
    if os.path.isfile(fOut):
        os.remove(fOut)

    data = Dataset(fOut, 'w')

    for dim in ['tile', 'nx', 'ny', 'nx1', 'ny1']:
        data.createDimension(dim, dimensions[dim])
    
    v0 = data.createVariable('lon', 'f8', ('tile', 'nx1', 'ny1'))
    v0[:] = variables['lon']
    v0.units = units['coord']
    v0 = data.createVariable('lat', 'f8', ('tile', 'nx1', 'ny1'))
    v0[:] = variables['lat']
    v0.units = units['coord']

    v1 = data.createVariable('ua', 'f8', ('tile', 'nx', 'ny'))
    v1[:] = variables['ua']
    v1.units = units['wind']
    v1 = data.createVariable('va', 'f8', ('tile', 'nx', 'ny'))
    v1[:] = variables['va']
    v1.units = units['wind']

    v1 = data.createVariable('psi', 'f8', ('tile', 'nx', 'ny'))
    v1[:] = variables['psi']
    v1.units = units['psi']

    v1 = data.createVariable('qvapor', 'f8', ('tile', 'nx', 'ny'))
    v1[:] = variables['qvapor']
    v1.units = units['qvapor']

    data.close()
    
    return


def unstagger_coord(field, mode='mean'):
    """
    Use: field = unstagger_coord(field, mode='mean')

    Unstaggers the coordinate that is +1 in length compared to the other. 

    Inputs: 
    - field: a staggered or unstaggered field
    - mode: mean (average of boundaries), first (first value only), last (last value only)

    Outputs:
    - field - unstaggered

    ** currently only works with fields that are the same size in x- and y- directions.
    ** TO DO: add final dimensions parameter to inputs and staggering based on that. 
    """
    fs = field.shape

    if len(fs) == 2:
        field = field[np.newaxis]
        zDim, dim1, dim2 = field.shape
    elif len(fs) == 3:
        zDim, dim1, dim2 = field.shape
    elif len(fs) == 4:
        zDim, dim1, dim2, dim3 = field.shape

    if mode == 'mean':
        if dim1 > dim2:
            field = 0.5 * (field[:, 1:, :] + field[:, :-1, :])
        elif dim2 > dim1:
            field = 0.5 * (field[:, :, 1:] + field[:, :, :-1])
        elif dim1 == dim2:
            pass
    
    elif mode == 'first':
        if dim1 > dim2:
            field = field[:, :-1, :]
        elif dim2 > dim1:
            field = field[:, :, :-1]
        elif dim1 == dim2:
            pass   
    
    elif mode == 'last':
        if dim1 > dim2:
            field = field[:, 1:, :]
        elif dim2 > dim1:
            field = field[:, :, 1:]
        elif dim1 == dim2:
            pass  
    
    if len(fs) == 2:
        field = field[0]

    return field


def plot_projection_field(lon, lat, field, cmap='viridis', vmin=-1, vmax=1, units='', title='', fSave=None):
    """
    Use: plot_projection_field(lon, lat, field, cmap='viridis', vmin=-1, vmax=1, units='', title='')

    Creates a Robinson projection and plots the (6) tiles on a map.

    Inputs:
    - lon, lat: lon and lat of coordinate edges (tile, x, y)
    - field: unstaggered field at a given vertical level (tile, x, y)
    - cmap: colormap
    - vmin, vmax: limits of the color map
    - units: label the units on the color bar
    - title: set title of the plot

    Outputs: none
    """


    fig = plt.figure(figsize = (8, 4))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    ax.set_facecolor('.4')

    f1 = pcolormesh_cube(lat, lon, field, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(f1, label=units)

    ax.set_title(title)

    if not fSave == None:
        plt.savefig(fSave, dpi=200, bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()

    return




