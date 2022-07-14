
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
from fv3viz import pcolormesh_cube



def unstagger_coord(field, mode='mean'):
    """
    mode options: 
    - mean: average of boundaries
    - first: leftmost value
    - last: rightmost value
    """
    if len(field.shape) == 3:
        zDim, dim1, dim2 = field.shape
    elif len(field.shape) == 4:
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

    return field






def plot_projection_field(lon, lat, field, cmap='viridis', vmin=-1, vmax=1, units='', title=''):



    fig = plt.figure(figsize = (8, 4))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection=ccrs.Robinson())
    ax.set_facecolor('.4')

    f1 = pcolormesh_cube(lat, lon, field, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(f1, label=units)

    ax.set_title(title)
    plt.show()


