.. meta::
   :robots: noindex, nofollow

=====
Usage
=====

Nudging
-------

Nudging functionality is provided by :py:func:`fv3gfs.util.apply_nudging` and
:py:func:`fv3gfs.util.get_nudging_tendencies`. The nudging tendencies can be stored to disk
by the user, for example using a :py:class:`fv3gfs.util.ZarrMonitor`. A runfile using this
functionality can be found in the `examples` directory.

Diagnostic IO
-------------

State can be persisted to disk using either :py:func:`fv3gfs.util.write_state` (described below)
or :py:class:`fv3gfs.util.ZarrMonitor`. The latter will coordinate between ranks to
write state to a unified Zarr store. Initializing it requires passing grid information.
This can be done directly from the namelist in a configuration dictionary like so::

    import fv3gfs.util
    from mpi4py import MPI
    import yaml

    with open('fv3config.yml', 'r') as f:
        config = yaml.safe_load(f)
    partitioner = fv3gfs.util.TilePartitioner.from_namelist(config['namelist'])

Alternatively, the grid information can be specified manually::

    partitioner = fv3gfs.util.TilePartitioner(
        layout=(1, 1)
    )

Once you have a :py:class:`fv3gfs.util.Partitioner`, the monitor can be created using any
Zarr store::

    import zarr
    store = zarr.storage.DirectoryStore('output_dir')  # relative or absolute path
    ZarrMonitor(partitioner, store, mode='w', mpi_comm=MPI.COMM_WORLD)

Note this can be used with any directory store available in ``zarr``.

Saving state to disk
--------------------

Sometimes you may want to write out model state to disk so that you can restart the model
from this state later. We provide a python-centric method for saving out and loading model state.
:py:func:`fv3gfs.util.read_state` saves the state on the current rank to a file on disk,
while :py:func:`fv3gfs.util.write_state` writes the rank's state to disk. Make sure you use
different filenames for each rank!

Loading Fortran Restarts
------------------------

A function :py:func:`fv3gfs.util.open_restart` is available to load restart files that have
been output by the Fortran FV3GFS model. This routine will handle
loading the data on a single processor per tile and then distribute the data to other
processes on the same tile. This may cause out-of-memory errors, which can be mitigated
in a couple different ways through changes to the code base (e.g. loading a subset of
the variables or levels at a time before distributing across ranks).

