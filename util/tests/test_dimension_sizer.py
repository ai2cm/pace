from collections import namedtuple

import pytest

import pace.util


@pytest.fixture(params=[48, 96])
def nx_tile(request):
    return request.param


@pytest.fixture(params=[48, 96])
def ny_tile(request, fast):
    if fast and request.param == 96:
        pytest.skip("running in fast mode")
    return request.param


@pytest.fixture(params=[60, 80])
def nz(request, fast):
    if fast and request.param == 80:
        pytest.skip("running in fast mode")
    return request.param


@pytest.fixture
def nx(nx_tile, layout):
    return nx_tile / layout[1]


@pytest.fixture
def ny(ny_tile, layout):
    return ny_tile / layout[0]


@pytest.fixture(params=[(1, 1), (3, 3)])
def layout(request):
    return request.param


@pytest.fixture
def extra_dimension_lengths():
    return {}


@pytest.fixture
def namelist(nx_tile, ny_tile, nz, layout):
    namelist = {
        "fv_core_nml": {
            "npx": nx_tile + 1,
            "npy": ny_tile + 1,
            "npz": nz,
            "layout": layout,
        }
    }
    return namelist


@pytest.fixture(params=["from_namelist", "from_tile_params"])
def sizer(request, nx_tile, ny_tile, nz, layout, namelist, extra_dimension_lengths):
    if request.param == "from_tile_params":
        sizer = pace.util.SubtileGridSizer.from_tile_params(
            nx_tile,
            ny_tile,
            nz,
            pace.util.N_HALO_DEFAULT,
            extra_dimension_lengths,
            layout,
        )
    elif request.param == "from_namelist":
        sizer = pace.util.SubtileGridSizer.from_namelist(namelist)
    else:
        raise NotImplementedError()
    return sizer


@pytest.fixture
def units():
    return "units_placeholder"


@pytest.fixture(params=[float, int])
def dtype(request):
    return request.param


DimCase = namedtuple("DimCase", ["dims", "origin", "extent", "shape"])


@pytest.fixture(
    params=[
        "x_only",
        "x_interface_only",
        "y_only",
        "y_interface_only",
        "z_only",
        "z_interface_only",
        "x_y",
        "z_y_x",
    ]
)
def dim_case(request, nx, ny, nz):
    if request.param == "x_only":
        return DimCase(
            (pace.util.X_DIM,),
            (pace.util.N_HALO_DEFAULT,),
            (nx,),
            (2 * pace.util.N_HALO_DEFAULT + nx + 1,),
        )
    elif request.param == "x_interface_only":
        return DimCase(
            (pace.util.X_INTERFACE_DIM,),
            (pace.util.N_HALO_DEFAULT,),
            (nx + 1,),
            (2 * pace.util.N_HALO_DEFAULT + nx + 1,),
        )
    elif request.param == "y_only":
        return DimCase(
            (pace.util.Y_DIM,),
            (pace.util.N_HALO_DEFAULT,),
            (ny,),
            (2 * pace.util.N_HALO_DEFAULT + ny + 1,),
        )
    elif request.param == "y_interface_only":
        return DimCase(
            (pace.util.Y_INTERFACE_DIM,),
            (pace.util.N_HALO_DEFAULT,),
            (ny + 1,),
            (2 * pace.util.N_HALO_DEFAULT + ny + 1,),
        )
    elif request.param == "z_only":
        return DimCase((pace.util.Z_DIM,), (0,), (nz,), (nz + 1,))
    elif request.param == "z_interface_only":
        return DimCase((pace.util.Z_INTERFACE_DIM,), (0,), (nz + 1,), (nz + 1,))
    elif request.param == "x_y":
        return DimCase(
            (
                pace.util.X_DIM,
                pace.util.Y_DIM,
            ),
            (pace.util.N_HALO_DEFAULT, pace.util.N_HALO_DEFAULT),
            (nx, ny),
            (
                2 * pace.util.N_HALO_DEFAULT + nx + 1,
                2 * pace.util.N_HALO_DEFAULT + ny + 1,
            ),
        )
    elif request.param == "z_y_x":
        return DimCase(
            (
                pace.util.Z_DIM,
                pace.util.Y_DIM,
                pace.util.X_DIM,
            ),
            (0, pace.util.N_HALO_DEFAULT, pace.util.N_HALO_DEFAULT),
            (nz, ny, nx),
            (
                nz + 1,
                2 * pace.util.N_HALO_DEFAULT + ny + 1,
                2 * pace.util.N_HALO_DEFAULT + nx + 1,
            ),
        )


@pytest.mark.cpu_only
def test_subtile_dimension_sizer_origin(sizer, dim_case):
    result = sizer.get_origin(dim_case.dims)
    assert result == dim_case.origin


@pytest.mark.cpu_only
def test_subtile_dimension_sizer_extent(sizer, dim_case):
    result = sizer.get_extent(dim_case.dims)
    assert result == dim_case.extent


@pytest.mark.cpu_only
def test_subtile_dimension_sizer_shape(sizer, dim_case):
    result = sizer.get_shape(dim_case.dims)
    assert result == dim_case.shape


def test_allocator_zeros(numpy, sizer, dim_case, units, dtype):
    allocator = pace.util.QuantityFactory(sizer, numpy)
    quantity = allocator.zeros(dim_case.dims, units, dtype=dtype)
    assert quantity.units == units
    assert quantity.dims == dim_case.dims
    assert quantity.origin == dim_case.origin
    assert quantity.extent == dim_case.extent
    assert quantity.data.shape == dim_case.shape
    assert numpy.all(quantity.data == 0)


def test_allocator_ones(numpy, sizer, dim_case, units, dtype):
    allocator = pace.util.QuantityFactory(sizer, numpy)
    quantity = allocator.ones(dim_case.dims, units, dtype=dtype)
    assert quantity.units == units
    assert quantity.dims == dim_case.dims
    assert quantity.origin == dim_case.origin
    assert quantity.extent == dim_case.extent
    assert quantity.data.shape == dim_case.shape
    assert numpy.all(quantity.data == 1)


def test_allocator_empty(numpy, sizer, dim_case, units, dtype):
    allocator = pace.util.QuantityFactory(sizer, numpy)
    quantity = allocator.empty(dim_case.dims, units, dtype=dtype)
    assert quantity.units == units
    assert quantity.dims == dim_case.dims
    assert quantity.origin == dim_case.origin
    assert quantity.extent == dim_case.extent
    assert quantity.data.shape == dim_case.shape
