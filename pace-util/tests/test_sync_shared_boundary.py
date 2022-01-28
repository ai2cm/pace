import pytest

import pace.util


@pytest.fixture
def dtype(numpy):
    return numpy.float64


@pytest.fixture
def units():
    return "m"


@pytest.fixture
def layout(request):
    try:
        return request.param
    except AttributeError:
        return (1, 1)


@pytest.fixture
def ranks_per_tile(layout):
    return layout[0] * layout[1]


@pytest.fixture
def total_ranks(ranks_per_tile):
    return 6 * ranks_per_tile


@pytest.fixture
def tile_partitioner(layout):
    return pace.util.TilePartitioner(layout)


@pytest.fixture
def cube_partitioner(tile_partitioner):
    return pace.util.CubedSpherePartitioner(tile_partitioner)


@pytest.fixture
def communicator_list(cube_partitioner, total_ranks):
    shared_buffer = {}
    return_list = []
    for rank in range(cube_partitioner.total_ranks):
        return_list.append(
            pace.util.CubedSphereCommunicator(
                comm=pace.util.testing.DummyComm(
                    rank=rank, total_ranks=total_ranks, buffer_dict=shared_buffer
                ),
                partitioner=cube_partitioner,
                timer=pace.util.Timer(),
            )
        )
    return return_list


@pytest.fixture
def rank_quantity_list(total_ranks, numpy, dtype, units=units):
    """
    Quantities whose values are equal to the rank
    """
    quantity_list = []
    for rank in range(total_ranks):
        x_data = numpy.empty((3, 2), dtype=dtype)
        x_data[:] = rank
        x_quantity = pace.util.Quantity(
            x_data,
            dims=(pace.util.Y_INTERFACE_DIM, pace.util.X_DIM),
            units=units,
            origin=(0, 0),
            extent=(3, 2),
        )
        y_data = numpy.empty((2, 3), dtype=dtype)
        y_data[:] = rank
        y_quantity = pace.util.Quantity(
            y_data,
            dims=(pace.util.Y_DIM, pace.util.X_INTERFACE_DIM),
            units=units,
            origin=(0, 0),
            extent=(2, 3),
        )
        quantity_list.append((x_quantity, y_quantity))
    return quantity_list


@pytest.fixture
def rank_target_list(total_ranks, numpy):
    return_list = []
    for rank in range(total_ranks):
        if rank % 2 == 0:
            target_x = (
                numpy.array([[rank, rank], [rank, rank], [rank + 2, rank + 2]]) % 6
            )
            target_y = numpy.array([[rank, rank, rank + 1], [rank, rank, rank + 1]]) % 6
        else:
            target_x = (
                numpy.array([[rank, rank], [rank, rank], [rank + 1, rank + 1]]) % 6
            )
            target_y = numpy.array([[rank, rank, rank + 2], [rank, rank, rank + 2]]) % 6
        return_list.append((target_x, target_y))
    return return_list


@pytest.mark.filterwarnings("ignore:invalid value encountered in remainder")
def test_correct_ranks_are_synchronized_with_no_halos(
    rank_quantity_list, communicator_list, subtests, numpy, rank_target_list
):
    req_list = []
    for communicator, (x_quantity, y_quantity) in zip(
        communicator_list, rank_quantity_list
    ):
        req = communicator.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        req_list.append(req)
    for req in req_list:
        req.wait()
    for (x_quantity, y_quantity), (target_x, target_y) in zip(
        rank_quantity_list, rank_target_list
    ):
        numpy.testing.assert_array_equal(numpy.abs(x_quantity.data), target_x)
        numpy.testing.assert_array_equal(numpy.abs(y_quantity.data), target_y)


@pytest.fixture
def counting_quantity_list(total_ranks, numpy, dtype, units=units):
    """
    A list of quantities whose entries increase sequentially in memory,
    with y values starting at 36 and x values starting at 0.
    """
    quantity_list = []
    for rank in range(total_ranks):
        x_data = numpy.array([[0, 1], [2, 3], [4, 5]]) + 6 * rank
        x_quantity = pace.util.Quantity(
            x_data,
            dims=(pace.util.Y_INTERFACE_DIM, pace.util.X_DIM),
            units=units,
            origin=(0, 0),
            extent=(3, 2),
        )
        y_data = 6 * total_ranks + numpy.array([[0, 1, 2], [3, 4, 5]]) + 6 * rank
        y_quantity = pace.util.Quantity(
            y_data,
            dims=(pace.util.Y_DIM, pace.util.X_INTERFACE_DIM),
            units=units,
            origin=(0, 0),
            extent=(2, 3),
        )
        quantity_list.append((x_quantity, y_quantity))
    return quantity_list


@pytest.mark.parametrize("layout", [(1, 1)], indirect=True)
def test_specific_edges_synced_correctly_on_first_rank(
    counting_quantity_list, communicator_list, subtests, numpy, rank_target_list
):
    """
    A test that a couple chosen edges send the correct data.

    Each example takes significant time to manually determine the correct answer,
    so this is limited to the first rank. Please add more cases as needed.
    """
    req_list = []
    for communicator, (x_quantity, y_quantity) in zip(
        communicator_list, counting_quantity_list
    ):
        req = communicator.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        req_list.append(req)
    for req in req_list:
        req.wait()
    first_rank_x, first_rank_y = counting_quantity_list[0]
    numpy.testing.assert_array_equal(
        first_rank_y.data, numpy.array([[36, 37, 42], [39, 40, 45]])
    )
    numpy.testing.assert_array_equal(
        first_rank_x.data, numpy.array([[0, 1], [2, 3], [-3 - 36 - 12, -36 - 12]])
    )
    second_rank_x, second_rank_y = counting_quantity_list[1]
    numpy.testing.assert_array_equal(
        second_rank_y.data, numpy.array([[42, 43, -19], [45, 46, -18]])
    )
    numpy.testing.assert_array_equal(
        second_rank_x.data, numpy.array([[6, 7], [8, 9], [12, 13]])
    )


@pytest.mark.parametrize("layout", [(3, 3)], indirect=True)
def test_interior_edges_synced_correctly_on_first_tile(
    counting_quantity_list,
    communicator_list,
    subtests,
    numpy,
    rank_target_list,
    total_ranks,
):
    """
    A test that a couple chosen edges send the correct data.

    Each example takes significant time to manually determine the correct answer,
    so this is limited to the first rank. Please add more cases as needed.
    """
    req_list = []
    for communicator, (x_quantity, y_quantity) in zip(
        communicator_list, counting_quantity_list
    ):
        req = communicator.start_synchronize_vector_interfaces(x_quantity, y_quantity)
        req_list.append(req)
    for req in req_list:
        req.wait()
    _, first_rank_y = counting_quantity_list[0]
    numpy.testing.assert_array_equal(
        first_rank_y.data, total_ranks * 6 + numpy.array([[0, 1, 6], [3, 4, 9]])
    )
    fifth_rank_x, fifth_rank_y = counting_quantity_list[4]
    numpy.testing.assert_array_equal(
        fifth_rank_y.data, (total_ranks + 4) * 6 + numpy.array([[0, 1, 6], [3, 4, 9]])
    )
    numpy.testing.assert_array_equal(
        fifth_rank_x.data, 4 * 6 + numpy.array([[0, 1], [2, 3], [18, 19]])
    )
