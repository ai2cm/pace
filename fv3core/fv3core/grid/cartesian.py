from pace.util.constants import PI


def setup_cartesian_grid(npx, npy, deglat, np):
    domain_rad = PI / 16.0
    lat_rad = deglat * PI / 180.0
    lon_rad = 0

    irange = np.arange(npx)
    jrange = np.arange(npy)

    longitudes = lon_rad - 0.5 * domain_rad + irange / (npx - 1) * domain_rad
    latitudes = lat_rad - 0.5 * domain_rad + jrange / (npx - 1) * domain_rad

    grid_longitudes = np.tile(longitudes, (npy, 1)).transpose()
    grid_latitudes = np.tile(latitudes, (npx, 1)).transpose()

    return grid_longitudes, grid_latitudes
