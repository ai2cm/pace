import copy
from typing import Callable, Sequence, Tuple

import numpy as np

from pace.dsl.stencil import GridIndexing


def get_subset_func(
    grid_indexing: GridIndexing,
    dims: Sequence[str],
    n_halo: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0)),
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Args:
        grid_indexing: configuration for domain and halo indexing
        dims: dimensions
        n_halo: number of halo points to retain

    Returns:
        subset: a subset function
    """
    origin, domain = grid_indexing.get_origin_domain(dims)
    i_start = origin[0] - n_halo[0][0]
    i_end = origin[0] + domain[0] + n_halo[0][1]
    j_start = origin[1] - n_halo[1][0]
    j_end = origin[1] + domain[1] + n_halo[1][1]

    def subset(data):
        if len(dims) == 3:
            return data[i_start:i_end, j_start:j_end, :]
        elif len(dims) == 2:
            return data[i_start:i_end, j_start:j_end]
        else:
            raise NotImplementedError(
                "Only 2D and 3D subsets are supported, got dims: {}".format(dims)
            )

    return subset


def get_set_nan_func(
    grid_indexing: GridIndexing,
    dims: Sequence[str],
    n_halo: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, 0), (0, 0)),
) -> Callable[[np.ndarray], np.ndarray]:
    subset = get_subset_func(grid_indexing=grid_indexing, dims=dims, n_halo=n_halo)

    def set_nans(data):
        try:
            safe = copy.deepcopy(data)
            data[:] = np.nan
            # data_subset is a view of data, so modifying data_subset modifies data
            data_subset = subset(data)
            data_subset[:] = subset(safe)
        except TypeError:
            safe = copy.deepcopy(data.storage)
            data.storage[:] = np.nan
            data_subset = subset(data.storage)
            data_subset[:] = subset(safe)

    return set_nans
