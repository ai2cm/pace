History
=======

latest
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
