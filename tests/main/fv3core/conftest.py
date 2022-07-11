import pace.dsl.stencil


mock_grid = pace.dsl.stencil.GridIndexing(
    domain=(12, 12, 79),
    n_halo=3,
    south_edge=True,
    north_edge=True,
    west_edge=True,
    east_edge=False,
)
mock_grid.rank = 0
