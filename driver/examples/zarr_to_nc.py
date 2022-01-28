import argparse

import xarray as xr
import zarr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts zarr directory stores to netcdf"
    )
    parser.add_argument("zarr_in", type=str, help="path of zarr to convert")
    parser.add_argument("netcdf_out", type=str, help="output netcdf")
    args = parser.parse_args()
    ds: xr.Dataset = xr.open_zarr(store=zarr.DirectoryStore(args.zarr_in))
    ds.to_netcdf(args.netcdf_out)
