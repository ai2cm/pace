from . import constants


def rotate_scalar_data(data, dims, numpy, n_clockwise_rotations):
    n_clockwise_rotations = n_clockwise_rotations % 4
    if n_clockwise_rotations == 0:
        pass
    elif n_clockwise_rotations in (1, 3):
        x_dim, y_dim = None, None
        for i, dim in enumerate(dims):
            if dim in constants.X_DIMS:
                x_dim = i
            elif dim in constants.Y_DIMS:
                y_dim = i
        if (x_dim is not None) and (y_dim is not None):
            if n_clockwise_rotations == 1:
                data = numpy.rot90(data, axes=(y_dim, x_dim))
            elif n_clockwise_rotations == 3:
                data = numpy.rot90(data, axes=(x_dim, y_dim))
        elif x_dim is not None:
            if n_clockwise_rotations == 1:
                data = numpy.flip(data, axis=x_dim)
        elif y_dim is not None:
            if n_clockwise_rotations == 3:
                data = numpy.flip(data, axis=y_dim)
    elif n_clockwise_rotations == 2:
        slice_list = []
        for dim in dims:
            if dim in constants.HORIZONTAL_DIMS:
                slice_list.append(slice(None, None, -1))
            else:
                slice_list.append(slice(None, None))
        data = data[tuple(slice_list)]
    return data


def rotate_vector_data(x_data, y_data, n_clockwise_rotations, dims, numpy):
    x_data = rotate_scalar_data(x_data, dims, numpy, n_clockwise_rotations)
    y_data = rotate_scalar_data(y_data, dims, numpy, n_clockwise_rotations)
    data = [x_data, y_data]
    n_clockwise_rotations = n_clockwise_rotations % 4
    if n_clockwise_rotations == 0:
        pass
    elif n_clockwise_rotations == 1:
        data[0], data[1] = data[1], -data[0]
    elif n_clockwise_rotations == 2:
        data[0], data[1] = -data[0], -data[1]
    elif n_clockwise_rotations == 3:
        data[0], data[1] = -data[1], data[0]
    return data
