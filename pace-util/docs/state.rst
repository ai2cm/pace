=====
State
=====

Quantity
--------

Data in pace-util is managed using a container type called :py:class:`pace.util.Quantity`.
This stores metadata such as dimensions and units (in ``quantity.dims`` and ``quantity.units``),
and manages the "computational domain" of the data.

When running a model on multiple
processors ("ranks"), such as in a cubed sphere configuration, each process is responsible for a subset of the domain, called its
"compute domain" or "computational domain". Arrays may contain additional data in a "halo" of "ghost cells"
which hold data from another rank's compute domain to be used as inputs for
the local rank. This data needs to be periodically retrieved from nearby ranks, as
the local rank cannot compute the new values outside of its compute domain.

A 3-by-3 array with one set of halo points would look something like::

    x x x x x
    x 0 0 0 x
    x 0 0 0 x
    x 0 0 0 x
    x x x x x

where `0` values represent the compute domain, and `x` represents points in the halo.
If you are interested in learning more, look up the "Ghost Cell Pattern" or
"Halo Exchange".

Depending on optimization choices, it may also make sense to include
filler data which serves only to align the computational domain into blocks within
memory.

If all of that sounded confusing, we agree! That's why :py:class:`pace.util.Quantity`
abstracts away as much of this information as possible. If you perform indexing on the
``view`` attribute of quantity, the index will be applied within the computational
domain::

    quantity.view[:] = 0.  # set all data this rank is responsible for to 0
    quantity.view[1:-1, :] = 1.0  # set data not on the first dimension edge to 1
    array = quantity.view[:]  # gives an array accessing just the compute domain
    new_array = np.copy(quantity.view[:])  # gives a copy of the compute domain

If you want to access data in ghost cells, instead of ``.view`` you should
access ``.data``, which is the underlying ``ndarray``-like object used by the ``Quantity``::

    quantity.data[:] = 0.  # set all data this rank has, including ghost cells, to zero
    quantity.data[quantity.origin[0]-3:quantity.origin[0]] == 1.  # set the left three ghost cells to 1
    array = quantity.data[quantity.origin[0]:quantity.origin[0]+quantity.extent[0]]  # same as quantity.view[:] for a 1D quantity

``data`` may be a numpy array or a cupy array. Both provide the same interface and
can be used identically. If you would like to use the appropriate "numpy" package
to manipulate your data, you can use ``quantity.np``. For example, the following
will give you the mean of your array, regardless of whether the data is on CPU or GPU,
and regardless of whether halo values are present::

    quantity.np.mean(quantity.view[:])
