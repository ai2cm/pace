import xarray as xr
import numpy as np

horizontal_names = ('x', 'y', 'x_interface', 'y_interface')


def without_ghost_cells(state, n_ghost):
    """Remove ghost cells from a state.

    Args:
        state (dict): a state dictionary with ghost cells
        n_ghost (int): number of ghost cells to remove

    Returns:
        state_without_ghost_cells (dict): a state dictionary whose DataArray objects point to
            the same underlying memory as the input state, but not the ghost cells.
    """
    state = state.copy()
    for name, value in state.items():
        if name == 'time':
            pass
        elif not isinstance(value, xr.DataArray):
            raise TypeError(f'value for {name} is of type {type(value)}, should be DataArray')
        else:
            dimension_count = value.ndim
            if dimension_count == 2:
                state[name] = value[n_ghost:-n_ghost, n_ghost:-n_ghost]
            elif dimension_count == 3:
                state[name] = value[:, n_ghost:-n_ghost, n_ghost:-n_ghost]
            elif dimension_count == 4:
                state[name] = value[:, :, n_ghost:-n_ghost, n_ghost:-n_ghost]
    return state


def pad_with_ghost_cells(data_array, n_ghost):
    array = data_array.values
    new_shape = list(array.shape)
    range_list = []
    for i, (dim_name, dim_length) in enumerate(zip(data_array.dims, data_array.shape)):
        if dim_name in horizontal_names:
            range_list.append(slice(n_ghost, n_ghost + dim_length))
            new_shape[i] += 2 * n_ghost
        else:
            range_list.append(slice(0, dim_length))
    new_array = np.empty(new_shape, dtype=array.dtype)
    new_array[tuple(range_list)] = array
    return xr.DataArray(
        new_array,
        dims=data_array.dims,
        attrs=data_array.attrs,
    )


def with_ghost_cells(state, n_ghost):
    """Add ghost cells to a state.
    
    Args:
        state (dict): a state dictionary without ghost cells
        n_ghost (int): number of ghost cells to add

    Returns:
        state_with_ghost_cells (dict): a copy of the state dictionary with ghost cells appended.
    """
    return_state = {}
    if 'time' in state:
        return_state['time'] = state['time']
    for name, data_array in state.items():
        if name != 'time':
            return_state[name] = pad_with_ghost_cells(data_array, n_ghost)
    return return_state
