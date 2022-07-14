
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube
from netCDF4 import Dataset
from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.stencil import GridIndexing, StencilConfig, StencilFactory
from pace.util import CubedSphereCommunicator, CubedSpherePartitioner, Quantity, QuantityFactory, SubtileGridSizer, TilePartitioner
from pace.util.constants import RADIUS
from pace.util.grid import DampingCoefficients, GridData, MetricTerms

import ipyparallel as ipp
import matplotlib.pyplot as plt
import numpy as np
import os



def show_clusters():
    """
    Use: show_clusters()

    Looks through the available clusters and prints their ids.
    
    Outputs: none
    """

    clusters = ipp.ClusterManager().load_clusters() 
    print("{:15} {:^10} {}".format("cluster_id", "state", "cluster_file")) 
    for c in clusters:
        cd = clusters[c].to_dict()
        cluster_id = cd['cluster']['cluster_id']
        controller_state = cd['controller']['state']['state']
        cluster_file = getattr(clusters[c], '_trait_values')['cluster_file']
        print("{:15} {:^10} {}".format(cluster_id, controller_state, cluster_file))
    
    return


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


    sizer = SubtileGridSizer.from_tile_params(nx_tile=dimensions['nx']-1, ny_tile=dimensions['ny']-1, nz=dimensions['nz'], n_halo=dimensions['nhalo'], 
                                              extra_dim_lengths={}, layout=layout, tile_partitioner=partitioner.tile, tile_rank=communicator.tile.rank)

    quantity_factory = QuantityFactory.from_backend(sizer=sizer, backend=backend)

    metric_terms = MetricTerms(quantity_factory=quantity_factory, communicator=communicator)
    damping_coefficients = DampingCoefficients.new_from_metric_terms(metric_terms)
    grid_data = GridData.new_from_metric_terms(metric_terms)

    dace_config = DaceConfig(communicator=None, backend=backend, orchestration=DaCeOrchestration.Python)
    stencil_config = StencilConfig(backend=backend, rebuild=False, validate_args=True, dace_config=dace_config)
    grid_indexing = GridIndexing.from_sizer_and_communicator(sizer=sizer, cube=communicator)
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


def create_gaussianMultiplier(lon, lat, dimensions, center_tile=0):
    """
    Use: gaussian_multiplier(lon, lat, dimensions, center_tile=0)

    Creates a 2-D Gaussian bell shape on the desired tile in the domain.
    RUNS ON RANK=0.

    Inputs:
    - lon: longitude of centerpoint (in radians)
    - lat: latitude of centerpoint (in radians)
    - dimensions: Dict{'nxhalo', 'nyhalo', 'tile'}
    - center_tile: the tile on which to center the blob; default = 0

    Outputs:
    - gaussian_multiplier: blob centered at the middle of center_tile with gaussian dropoff
    """
    from pace.util.grid.gnomonic import great_circle_distance_lon_lat

    gaussian_multiplier = np.zeros((dimensions['tile'], dimensions['nxhalo'], dimensions['nyhalo']))
    r0 = RADIUS / 3.

    p1x, p1y = int(dimensions['nxhalo']/2), int(dimensions['nyhalo']/2) # center gaussian on middle of tile
    p_center = [lon[center_tile, p1x, p1y], lat[center_tile, p1x, p1y]]
    print('Centering gaussian on lon=%.2f, lat=%.2f' % (np.rad2deg(p_center[0]), np.rad2deg(p_center[1])))

    for tile in range(dimensions['tile']):
        for jj in range(dimensions['nyhalo']):
            for ii in range(dimensions['nxhalo']):
                p_dist = [lon[tile, ii, jj], lat[tile, ii, jj]]

                r = great_circle_distance_lon_lat(p_center[0], p_dist[0], p_center[1], p_dist[1], RADIUS, np) 

                if tile == center_tile:
                    gaussian_multiplier[tile, ii, jj] = 0.5 * (1.0 + np.cos(np.pi * r / r0)) if r < r0 else 0.0

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

    Ubar = (2.0 * np.pi * RADIUS) / (12. * 86400.0)  # 38.6 
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


def calculate_windsFromStreamfunction(psi, dx, dy, dimensions):
    """
    Use: ua, va = calculate_windsFromStreamfunction(psi, dx, dy, dimensions)

    Inputs:
    - psi: streamfunction on center points; with halo points
    - dx, dy: distance between center points; with halo points
    - dimensions: Dict{'nxhalo', 'nyhalo', 'nx', 'ny}
    """
    ua = np.zeros((dimensions['nxhalo'], dimensions['nyhalo']))
    va = np.zeros((dimensions['nxhalo'], dimensions['nyhalo']))

    for jj in range(dimensions['nhalo']-1, dimensions['ny']+dimensions['nhalo']+1):
        for ii in range(dimensions['nhalo']-1, dimensions['nx']+dimensions['nhalo']+1):
            psi1 = 0.5 * (psi.data[ii, jj] + psi.data[ii, jj-1])
            psi2 = 0.5 * (psi.data[ii, jj] + psi.data[ii, jj+1])
            dist = dy.data[ii, jj]
            ua[ii, jj] = 0 if dist == 0 else -1.0 * (psi2 - psi1) / dist

            psi1 = 0.5 * (psi.data[ii, jj] + psi.data[ii-1, jj])
            psi2 = 0.5 * (psi.data[ii, jj] + psi.data[ii+1, jj])
            dist = dx.data[ii, jj]
            va[ii, jj] = 0 if dist == 0 else (psi2 - psi1) / dist

    return ua, va


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


def plot_projection_field(lon, lat, field, cmap='viridis', vmin=-1, vmax=1, units='', title=''):
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
    plt.show()
    return


