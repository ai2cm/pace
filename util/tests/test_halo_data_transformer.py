import copy
from typing import Tuple

import numpy as np
import pytest

from pace.util import (
    EAST,
    NORTH,
    NORTHEAST,
    NORTHWEST,
    SOUTH,
    SOUTHEAST,
    SOUTHWEST,
    WEST,
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
    Quantity,
    _boundary_utils,
)
from pace.util.buffer import Buffer
from pace.util.halo_data_transformer import (
    HaloDataTransformer,
    HaloExchangeSpec,
    QuantityHaloSpec,
)
from pace.util.rotate import rotate_scalar_data, rotate_vector_data


@pytest.fixture
def nz():
    return 5


@pytest.fixture
def ny():
    return 7


@pytest.fixture
def nx():
    return 7


@pytest.fixture
def units():
    return "m"


@pytest.fixture(params=[0, 1])
def n_buffer(request):
    return request.param


@pytest.fixture
def n_points():
    return 1


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture(params=[1, 3])
def n_halos(request):
    return request.param


@pytest.fixture
def origin(n_halos, dims, n_buffer):
    return_list = []
    origin_dict = {
        X_DIM: n_halos + n_buffer,
        X_INTERFACE_DIM: n_halos + n_buffer,
        Y_DIM: n_halos + n_buffer,
        Y_INTERFACE_DIM: n_halos + n_buffer,
        Z_DIM: n_buffer,
        Z_INTERFACE_DIM: n_buffer,
    }
    for dim in dims:
        return_list.append(origin_dict[dim])
    return return_list


@pytest.fixture(
    params=[
        pytest.param((Y_DIM, X_DIM), id="center"),
        pytest.param((Z_DIM, Y_DIM, X_DIM), id="center_3d"),
        pytest.param(
            (X_DIM, Y_DIM, Z_DIM),
            id="center_3d_reverse",
        ),
        pytest.param(
            (X_DIM, Z_DIM, Y_DIM),
            id="center_3d_shuffle",
        ),
        pytest.param((Y_INTERFACE_DIM, X_INTERFACE_DIM), id="interface"),
        pytest.param(
            (
                Z_INTERFACE_DIM,
                Y_INTERFACE_DIM,
                X_INTERFACE_DIM,
            ),
            id="interface_3d",
        ),
    ]
)
def dims(request, fast):
    if fast and request.param in (
        (X_DIM, Y_DIM, Z_DIM),
        (
            Z_INTERFACE_DIM,
            Y_INTERFACE_DIM,
            X_INTERFACE_DIM,
        ),
    ):
        pytest.skip("running in fast mode")
    return request.param


@pytest.fixture
def shape(nz, ny, nx, dims, n_halos, n_buffer):
    return_list = []
    length_dict = {
        X_DIM: 2 * n_halos + nx + n_buffer,
        X_INTERFACE_DIM: 2 * n_halos + nx + 1 + n_buffer,
        Y_DIM: 2 * n_halos + ny + n_buffer,
        Y_INTERFACE_DIM: 2 * n_halos + ny + 1 + n_buffer,
        Z_DIM: nz + n_buffer,
        Z_INTERFACE_DIM: nz + 1 + n_buffer,
    }
    for dim in dims:
        return_list.append(length_dict[dim])
    return return_list


@pytest.fixture
def extent(n_points, dims, nz, ny, nx):
    return_list = []
    extent_dict = {
        X_DIM: nx,
        X_INTERFACE_DIM: nx + 1,
        Y_DIM: ny,
        Y_INTERFACE_DIM: ny + 1,
        Z_DIM: nz,
        Z_INTERFACE_DIM: nz + 1,
    }
    for dim in dims:
        return_list.append(extent_dict[dim])
    return return_list


def _shape_length(shape: Tuple[int]) -> int:
    """Compute linear size from slices"""
    length = 1
    for s in shape:
        length *= s
    return length


@pytest.fixture
def quantity(dims, units, origin, extent, shape, dtype, gt4py_backend):
    """A list of quantities whose values are 42.42 in the computational domain and 1
    outside of it."""
    sz = _shape_length(shape)
    print(f"{shape} {sz}")
    data = np.arange(0, sz, dtype=dtype).reshape(shape)
    if "gtc" not in gt4py_backend:
        # should also test code if gt4py_backend is unset
        gt4py_backend = None
    quantity = Quantity(
        data,
        dims=dims,
        units=units,
        origin=origin,
        extent=extent,
        gt4py_backend=gt4py_backend,
    )
    return quantity


@pytest.fixture(params=[-0, -1, -2, -3])
def rotation(request):
    return request.param


def test_data_transformer_allocate(quantity, n_halos):
    boundary_north = _boundary_utils.get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        NORTH,
        n_halos,
        interior=False,
    )
    boundary_southwest = _boundary_utils.get_boundary_slice(
        quantity.dims,
        quantity.origin,
        quantity.extent,
        quantity.data.shape,
        SOUTHWEST,
        n_halos,
        interior=False,
    )

    specification = QuantityHaloSpec(
        n_points=n_halos,
        shape=quantity.data.shape,
        strides=quantity.data.strides,
        itemsize=quantity.data.itemsize,
        origin=quantity.metadata.origin,
        extent=quantity.metadata.extent,
        dims=quantity.metadata.dims,
        numpy_module=quantity.np,
        dtype=quantity.metadata.dtype,
    )

    exchange_descriptors = [
        HaloExchangeSpec(specification, boundary_north, 0, boundary_north),
        HaloExchangeSpec(specification, boundary_southwest, 0, boundary_southwest),
    ]

    data_transformer = HaloDataTransformer.get(quantity.np, exchange_descriptors)

    assert len(data_transformer.get_pack_buffer().array.shape) == 1
    assert (
        data_transformer.get_pack_buffer().array.size
        == quantity.data[boundary_north].size + quantity.data[boundary_southwest].size
    )
    assert len(data_transformer.get_unpack_buffer().array.shape) == 1
    assert (
        data_transformer.get_unpack_buffer().array.size
        == quantity.data[boundary_north].size + quantity.data[boundary_southwest].size
    )
    # clean up
    Buffer.push_to_cache(data_transformer._pack_buffer)
    Buffer.push_to_cache(data_transformer._unpack_buffer)


def _get_boundaries(quantity, n_halos):
    send_boundaries = {}
    recv_boundaries = {}
    for direction in [
        NORTH,
        NORTHWEST,
        WEST,
        SOUTHWEST,
        SOUTH,
        SOUTHEAST,
        EAST,
        NORTHEAST,
    ]:
        send_boundaries[direction] = _boundary_utils.get_boundary_slice(
            quantity.dims,
            quantity.origin,
            quantity.extent,
            quantity.data.shape,
            direction,
            n_halos,
            interior=True,
        )
        recv_boundaries[direction] = _boundary_utils.get_boundary_slice(
            quantity.dims,
            quantity.origin,
            quantity.extent,
            quantity.data.shape,
            direction,
            n_halos,
            interior=False,
        )

    return send_boundaries, recv_boundaries


def test_data_transformer_scalar_pack_unpack(quantity, rotation, n_halos):
    target_quantity: Quantity = copy.deepcopy(quantity)

    send_boundaries, recv_boundaries = _get_boundaries(quantity, n_halos)

    NE_corner_boundaries = {
        0: (send_boundaries[NORTHEAST], recv_boundaries[SOUTHEAST]),
        -1: (send_boundaries[NORTHEAST], recv_boundaries[SOUTHWEST]),
        -2: (send_boundaries[NORTHEAST], recv_boundaries[NORTHWEST]),
        -3: (send_boundaries[NORTHEAST], recv_boundaries[NORTHEAST]),
    }

    N_edge_boundaries = {
        0: (send_boundaries[NORTH], recv_boundaries[SOUTH]),
        -1: (send_boundaries[NORTH], recv_boundaries[WEST]),
        -2: (send_boundaries[NORTH], recv_boundaries[NORTH]),
        -3: (send_boundaries[NORTH], recv_boundaries[EAST]),
    }

    specification = QuantityHaloSpec(
        n_points=n_halos,
        shape=quantity.data.shape,
        strides=quantity.data.strides,
        itemsize=quantity.data.itemsize,
        origin=quantity.metadata.origin,
        extent=quantity.metadata.extent,
        dims=quantity.metadata.dims,
        numpy_module=quantity.np,
        dtype=quantity.metadata.dtype,
    )

    exchange_descriptors = [
        HaloExchangeSpec(
            specification,
            N_edge_boundaries[rotation][0],
            rotation,
            N_edge_boundaries[rotation][1],
        ),
        HaloExchangeSpec(
            specification,
            NE_corner_boundaries[rotation][0],
            rotation,
            NE_corner_boundaries[rotation][1],
        ),
    ]

    data_transformer = HaloDataTransformer.get(quantity.np, exchange_descriptors)

    data_transformer.async_pack([quantity, quantity])
    # Simulate data transfer
    data_transformer.get_unpack_buffer().assign_from(
        data_transformer.get_pack_buffer().array
    )
    data_transformer.async_unpack([quantity, quantity])
    data_transformer.synchronize()

    # From the copy of the original quantity we rotate data
    # according to the rotation & slice and insert them back
    # this reproduces the multi-buffer strategy
    rotated = rotate_scalar_data(
        quantity.data[N_edge_boundaries[rotation][0]],
        quantity.dims,
        quantity.metadata.np,
        -rotation,
    )
    target_quantity.data[N_edge_boundaries[rotation][1]] = rotated
    rotated = rotate_scalar_data(
        quantity.data[NE_corner_boundaries[rotation][0]],
        quantity.dims,
        quantity.metadata.np,
        -rotation,
    )
    target_quantity.data[NE_corner_boundaries[rotation][1]] = rotated

    assert (target_quantity.data == quantity.data).all()


def test_data_transformer_vector_pack_unpack(quantity, rotation, n_halos):
    targe_quanity_x = copy.deepcopy(quantity)
    targe_quanity_y = copy.deepcopy(targe_quanity_x)
    x_quantity = quantity
    y_quantity = copy.deepcopy(x_quantity)

    send_boundaries, recv_boundaries = _get_boundaries(x_quantity, n_halos)

    NE_corner_boundaries = {
        0: (send_boundaries[NORTHEAST], recv_boundaries[SOUTHEAST]),
        -1: (send_boundaries[NORTHEAST], recv_boundaries[SOUTHWEST]),
        -2: (send_boundaries[NORTHEAST], recv_boundaries[NORTHWEST]),
        -3: (send_boundaries[NORTHEAST], recv_boundaries[NORTHEAST]),
    }

    N_edge_boundaries = {
        0: (send_boundaries[NORTH], recv_boundaries[SOUTH]),
        -1: (send_boundaries[NORTH], recv_boundaries[WEST]),
        -2: (send_boundaries[NORTH], recv_boundaries[NORTH]),
        -3: (send_boundaries[NORTH], recv_boundaries[EAST]),
    }

    specification_x = QuantityHaloSpec(
        n_points=n_halos,
        shape=x_quantity.data.shape,
        strides=x_quantity.data.strides,
        itemsize=x_quantity.data.itemsize,
        origin=x_quantity.metadata.origin,
        extent=x_quantity.metadata.extent,
        dims=x_quantity.metadata.dims,
        numpy_module=x_quantity.np,
        dtype=x_quantity.metadata.dtype,
    )
    specification_y = QuantityHaloSpec(
        n_points=n_halos,
        shape=y_quantity.data.shape,
        strides=y_quantity.data.strides,
        itemsize=y_quantity.data.itemsize,
        origin=y_quantity.metadata.origin,
        extent=y_quantity.metadata.extent,
        dims=y_quantity.metadata.dims,
        numpy_module=y_quantity.np,
        dtype=y_quantity.metadata.dtype,
    )

    exchange_descriptors_x = [
        HaloExchangeSpec(
            specification_x,
            N_edge_boundaries[rotation][0],
            rotation,
            N_edge_boundaries[rotation][1],
        ),
        HaloExchangeSpec(
            specification_x,
            NE_corner_boundaries[rotation][0],
            rotation,
            NE_corner_boundaries[rotation][1],
        ),
    ]
    exchange_descriptors_y = [
        HaloExchangeSpec(
            specification_y,
            N_edge_boundaries[rotation][0],
            rotation,
            N_edge_boundaries[rotation][1],
        ),
        HaloExchangeSpec(
            specification_y,
            NE_corner_boundaries[rotation][0],
            rotation,
            NE_corner_boundaries[rotation][1],
        ),
    ]

    data_transformer = HaloDataTransformer.get(
        x_quantity.np, exchange_descriptors_x, exchange_descriptors_y
    )

    data_transformer.async_pack([x_quantity, x_quantity], [y_quantity, y_quantity])
    # Simulate data transfer
    data_transformer.get_unpack_buffer().assign_from(
        data_transformer.get_pack_buffer().array
    )
    data_transformer.async_unpack([x_quantity, x_quantity], [y_quantity, y_quantity])
    data_transformer.synchronize()

    # From the copy of the original quantity we rotate data
    # according to the rotation & slice and insert them bak
    # this reproduce the multi-buffer strategy
    rotated_x, rotated_y = rotate_vector_data(
        quantity.data[N_edge_boundaries[rotation][0]],
        quantity.data[N_edge_boundaries[rotation][0]],
        -rotation,
        quantity.dims,
        quantity.metadata.np,
    )
    targe_quanity_x.data[N_edge_boundaries[rotation][1]] = rotated_x
    targe_quanity_y.data[N_edge_boundaries[rotation][1]] = rotated_y
    rotated_x, rotated_y = rotate_vector_data(
        quantity.data[NE_corner_boundaries[rotation][0]],
        quantity.data[NE_corner_boundaries[rotation][0]],
        -rotation,
        quantity.dims,
        quantity.metadata.np,
    )
    targe_quanity_x.data[NE_corner_boundaries[rotation][1]] = rotated_x
    targe_quanity_y.data[NE_corner_boundaries[rotation][1]] = rotated_y

    assert (targe_quanity_x.data == x_quantity.data).all()
    assert (targe_quanity_y.data == y_quantity.data).all()
