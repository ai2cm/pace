History
=======

v0.7.0
------

Major changes:
- Renamed package from fv3gfs-util to pace-util
- Added NullTimer to use for default Timer value, it is a disabled timer which cannot be enabled (raises NotImplementedError)
- Added pace.util.grid, keeping symbols out of top level as they are still unstable
- Added HaloUpdater and associated code, which compiles halo packing for more efficient halo updates
- Added physical constants to pace.util.constants

Minor changes:
- Added method set_extra_dim_lengths to QuantityFactory

Fixes:
- Fixed bug where ZarrMonitor depended on dict `.items()` always returning items in the same order

Other changes may exist in this version, as we temporarily paused updating the history on each PR.

v0.6.0
------

Major changes:
- Use `cftime.datetime` objects to represent datetimes instead
of `datetime.datetime` objects.  This results in times stored in a format compatible with
the fortran model, and accurate internal representation of times with the calendar specified
in the `coupler_nml` namelist.
- `Timer` class is added, with methods `start` and `stop`, and properties `clock` (context manager), and `times` (dictionary of accumulated timing)
- `CubedSphereCommunicator` instances now have a `.timer` attribute, which accumulates times for "pack", "unpack", "Isend", and "Recv" during halo updates
- make `SubtileGridSizer.from_tile_params` public API
- New method `CubedSphereCommunicator.synchronize_vector_interfaces` which synchronizes edge values on interface variables which are duplicated between adjacent ranks
- Added `.sel` method to corner views (e.g. `quantity.view.northeast.sel(x=0, y=1)`) to allow indexing these corner views with arbitrary dimension ordering.
- Halo updates now use tagged send/recv operations, which prevents deadlocks in certain situations
- Quantity.data is now guaranteed to be a numpy or cupy array matching its `.np` module, and will no longer be a gt4py Storage
- Quantity accepts a `gt4py_backend` on initialize which is used to create its `.storage` if one was not used on initialize
- parent MPI rank now referred to as "root" rank in variable names and documentation
- Added TILE_DIM constant for tile dimension of global quantities
- Added Partitioner base class implementing features necessary for scatter/gather
- Moved scatter and gather from TileCommunicator to the Communicator base class, so its code can be re-used by the CubedSphereCommunicator
- Implemented subtile_slice, global_extent, and subtile_extent routines on CubedSpherePartitioner necessary for scatter/gather in CubedSphereCommunicator
- Renamed argument `tile_extent` and `tile_dims` to `global_extent` and `global_dims` in routines to refer generically to the tile in the case of tile scatter/gather or cube in the case of cube scatter/gather
- Fixed a bug where initializing a Quantity with a numpy array and a gpu backend would give CPUStorage
- raise TypeError if initializing a quantity with both a storage and a gt4py_backend argument
- eagerly create storage object when initializing Quantity
- make data type of quantity and storage reflect the gt4py_backend chosen, instead of being determined based on the data type being numpy/cupy

Fixes:
- If `only_names` is provided to `open_restart`, it will return those fields and nothing more.  Previously it would include `"time"` in the returned state even if it was not requested.
- Fixed a bug where quantity.storage and quantity.data could be out of sync if the quantity was initialized using data and a gt4py backend string
- Default slice for corner views when not given at all as an index (e.g. when providing one index to a 2D view) now gives the same result as providing an empty slice (:)
- Fixed a bug where quantity.view could refer to a different array than quantity.data if the quantity was initialized using data and a gt4py backend string, and then quantity.storage was accessed

v0.5.1
------

- enable MPI tests on CircleCI

v0.5.0
------

Breaking changes:
- `send_buffer` and `recv_buffer` are modified to take in a `callable`, which is more easily serialized than a `numpy`-like module (necessary because we serialize the arguments to re-use buffers), and allows custom specification of the initialization if zeros are needed instead of empty.

Major changes:
- Added additional regional views to Quantity as attributes on Quantity.view, including `northeast`, `northwest`, `southeast`, `southwest`, and `interior`
- Separated fv3util into its own repository and began tracking history separately from fv3gfs-python
- Added getters and setters for additional dynamics quantities needed to call an alternative dynamical core
- Added `storage` property to Quantity, implemented as short-term shortcut to .data until gt4py GDP-3 is implemented

Deprecations:
- `Quantity.values` is deprecated

v0.4.3 (2020-05-15)
-------------------

Last release of fv3util with history contained in fv3gfs-python.
