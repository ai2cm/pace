import fv3core
import fv3core.utils.grid


mock_grid = fv3core.utils.grid.GridIndexing(
    domain=(12, 12, 79),
    n_halo=3,
    south_edge=True,
    north_edge=True,
    west_edge=True,
    east_edge=False,
)
mock_grid.rank = 0

fv3core._config.grid = mock_grid
