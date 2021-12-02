import numpy as np
import pytest

import pace.util


@pytest.fixture
def quantity(request):
    return pace.util.Quantity(
        request.param[0],
        dims=request.param[1],
        units="units",
    )


# edge views were implemented but not enabled, since the API is not yet needed and
# might be subject to change - tests are included here as comments


# @pytest.mark.parametrize(
#     "quantity, view_slice, reference",
#     [
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (0, 0),
#             4,
#             id="3_by_3_center_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([1, 2, 3]),
#                 dims=[pace.util.X_DIM],
#                 units="m",
#                 origin=(1,),
#                 extent=(1,),
#             ),
#             (-1,),
#             1,
#             id="3_1d_left_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(0, 1), slice(None, None)),
#             np.array([[4]]),
#             id="3_by_3_center_value_as_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (-1, 0),
#             1,
#             id="3_by_3_first_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(-1, 0), slice(0, 1)),
#             np.array([[1]]),
#             id="3_by_3_first_value_as_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(None, None), slice(None, None)),
#             np.array([[4]]),
#             id="3_by_3_default_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(2, 2),
#                 extent=(1, 1),
#             ),
#             (slice(None, None), slice(None, None)),
#             np.array([[12]]),
#             id="5_by_5_mostly_halo_default_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(2, 2),
#                 extent=(1, 1),
#             ),
#             (slice(-2, 0), slice(None, None)),
#             np.array([[2], [7]]),
#             id="5_by_5_larger_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(3, 2),
#                 extent=(1, 1),
#             ),
#             (slice(-3, 0), slice(None, None)),
#             np.array([[2], [7], [12]]),
#             id="5_by_5_shifted_right_larger_slice",
#         ),
#     ],
# )
# def test_west(quantity, view_slice, reference):
#     result = quantity.view.west[view_slice]
#     quantity.np.testing.assert_array_equal(result, reference)
#     # result should be a slice of the quantity memory, if it's a slice
#     assert len(result.shape) == 0 or result.base is quantity.data
#     transposed_quantity = pace.util.Quantity(
#         quantity.data.T,
#         dims=quantity.dims[::-1],
#         units=quantity.units,
#         origin=quantity.origin[::-1],
#         extent=quantity.extent[::-1],
#     )
#     transposed_result = transposed_quantity.view.west[view_slice[::-1]]
#     if isinstance(reference, quantity.np.ndarray):
#         quantity.np.testing.assert_array_equal(transposed_result, reference.T)
#     else:
#         quantity.np.testing.assert_array_equal(transposed_result, reference)


@pytest.mark.parametrize(
    "quantity",
    [
        pace.util.Quantity(
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            dims=[pace.util.X_DIM, pace.util.Y_DIM],
            units="m",
            origin=(1, 1),
            extent=(1, 1),
        )
    ],
)
@pytest.mark.parametrize(
    "view_name",
    [
        # "east",
        # "west",
        # "north",
        # "south",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
        "interior",
    ],
)
def test_many_indices_raises(quantity, view_name):
    view = getattr(quantity.view, view_name)
    index = tuple([0] * (len(quantity.dims) + 1))
    with pytest.raises(IndexError):
        view[index]


@pytest.mark.parametrize(
    "quantity",
    [
        pace.util.Quantity(
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            dims=[pace.util.X_DIM, pace.util.Y_DIM],
            units="m",
            origin=(1, 1),
            extent=(1, 1),
        )
    ],
)
@pytest.mark.parametrize(
    "view_name",
    [
        # "east",
        # "west",
        # "north",
        # "south",
        "northeast",
        "northwest",
        "southeast",
        "southwest",
        "interior",
    ],
)
def test_many_slices_raises(quantity, view_name):
    view = getattr(quantity.view, view_name)
    index = tuple([slice(0, 1)] * (len(quantity.dims) + 1))
    with pytest.raises(IndexError):
        view[index]


# @pytest.mark.parametrize(
#     "quantity, view_slice, reference",
#     [
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (-1, 0),
#             4,
#             id="3_by_3_center_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([1, 2, 3]),
#                 dims=[pace.util.X_DIM],
#                 units="m",
#                 origin=(1,),
#                 extent=(1,),
#             ),
#             (0,),
#             3,
#             id="3_1d_right_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(-1, 0), slice(None, None)),
#             np.array([[4]]),
#             id="3_by_3_center_value_as_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (0, 0),
#             7,
#             id="3_by_3_first_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(0, 1), slice(0, 1)),
#             np.array([[7]]),
#             id="3_by_3_first_value_as_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(None, None), slice(None, None)),
#             np.array([[4]]),
#             id="3_by_3_default_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(2, 2),
#                 extent=(1, 1),
#             ),
#             (slice(None, None), slice(None, None)),
#             np.array([[12]]),
#             id="5_by_5_mostly_halo_default_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(2, 2),
#                 extent=(1, 1),
#             ),
#             (slice(0, 2), slice(None, None)),
#             np.array([[17], [22]]),
#             id="5_by_5_larger_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.X_DIM, pace.util.Y_DIM],
#                 units="m",
#                 origin=(1, 2),
#                 extent=(1, 1),
#             ),
#             (slice(0, 3), slice(None, None)),
#             np.array([[12], [17], [22]]),
#             id="5_by_5_shifted_left_larger_slice",
#         ),
#     ],
# )
# def test_east(quantity, view_slice, reference):
#     result = quantity.view.east[view_slice]
#     quantity.np.testing.assert_array_equal(result, reference)
#     # result should be a slice of the quantity memory, if it's a slice
#     assert len(result.shape) == 0 or result.base is quantity.data
#     transposed_quantity = pace.util.Quantity(
#         quantity.data.T,
#         dims=quantity.dims[::-1],
#         units=quantity.units,
#         origin=quantity.origin[::-1],
#         extent=quantity.extent[::-1],
#     )
#     transposed_result = transposed_quantity.view.east[view_slice[::-1]]
#     if isinstance(reference, quantity.np.ndarray):
#         quantity.np.testing.assert_array_equal(transposed_result, reference.T)
#     else:
#         quantity.np.testing.assert_array_equal(transposed_result, reference)


# @pytest.mark.parametrize(
#     "quantity, view_slice, reference",
#     [
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (0, 0),
#             4,
#             id="3_by_3_center_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([1, 2, 3]),
#                 dims=[pace.util.Y_DIM],
#                 units="m",
#                 origin=(1,),
#                 extent=(1,),
#             ),
#             (-1,),
#             1,
#             id="3_1d_bottom_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([1, 2, 3]),
#                 dims=[pace.util.Y_INTERFACE_DIM],
#                 units="m",
#                 origin=(1,),
#                 extent=(1,),
#             ),
#             (-1,),
#             1,
#             id="3_1d_bottom_interface_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(0, 1), slice(None, None)),
#             np.array([[4]]),
#             id="3_by_3_center_value_as_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (-1, 0),
#             1,
#             id="3_by_3_first_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(-1, 0), slice(0, 1)),
#             np.array([[1]]),
#             id="3_by_3_first_value_as_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(None, None), slice(None, None)),
#             np.array([[4]]),
#             id="3_by_3_default_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(2, 2),
#                 extent=(1, 1),
#             ),
#             (slice(None, None), slice(None, None)),
#             np.array([[12]]),
#             id="5_by_5_mostly_halo_default_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(2, 2),
#                 extent=(1, 1),
#             ),
#             (slice(-2, 0), slice(None, None)),
#             np.array([[2], [7]]),
#             id="5_by_5_larger_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(3, 2),
#                 extent=(1, 1),
#             ),
#             (slice(-3, 0), slice(None, None)),
#             np.array([[2], [7], [12]]),
#             id="5_by_5_shifted_larger_slice",
#         ),
#     ],
# )
# def test_south(quantity, view_slice, reference):
#     result = quantity.view.south[view_slice]
#     quantity.np.testing.assert_array_equal(result, reference)
#     # result should be a slice of the quantity memory, if it's a slice
#     assert len(result.shape) == 0 or result.base is quantity.data
#     transposed_quantity = pace.util.Quantity(
#         quantity.data.T,
#         dims=quantity.dims[::-1],
#         units=quantity.units,
#         origin=quantity.origin[::-1],
#         extent=quantity.extent[::-1],
#     )
#     transposed_result = transposed_quantity.view.south[view_slice[::-1]]
#     if isinstance(reference, quantity.np.ndarray):
#         quantity.np.testing.assert_array_equal(transposed_result, reference.T)
#     else:
#         quantity.np.testing.assert_array_equal(transposed_result, reference)


# @pytest.mark.parametrize(
#     "quantity, view_slice, reference",
#     [
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (-1, 0),
#             4,
#             id="3_by_3_center_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([1, 2, 3]),
#                 dims=[pace.util.Y_DIM],
#                 units="m",
#                 origin=(1,),
#                 extent=(1,),
#             ),
#             (0,),
#             3,
#             id="3_1d_top_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([1, 2, 3]),
#                 dims=[pace.util.Y_INTERFACE_DIM],
#                 units="m",
#                 origin=(1,),
#                 extent=(1,),
#             ),
#             (0,),
#             3,
#             id="3_1d_top_interface_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(-1, 0), slice(None, None)),
#             np.array([[4]]),
#             id="3_by_3_center_value_as_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (0, 0),
#             7,
#             id="3_by_3_first_value",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(0, 1), slice(0, 1)),
#             np.array([[7]]),
#             id="3_by_3_first_value_as_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 1),
#                 extent=(1, 1),
#             ),
#             (slice(None, None), slice(None, None)),
#             np.array([[4]]),
#             id="3_by_3_default_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(2, 2),
#                 extent=(1, 1),
#             ),
#             (slice(None, None), slice(None, None)),
#             np.array([[12]]),
#             id="5_by_5_mostly_halo_default_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(2, 2),
#                 extent=(1, 1),
#             ),
#             (slice(0, 2), slice(None, None)),
#             np.array([[17], [22]]),
#             id="5_by_5_larger_slice",
#         ),
#         pytest.param(
#             pace.util.Quantity(
#                 np.array(
#                     [
#                         [0, 1, 2, 3, 4],
#                         [5, 6, 7, 8, 9],
#                         [10, 11, 12, 13, 14],
#                         [15, 16, 17, 18, 19],
#                         [20, 21, 22, 23, 24],
#                     ]
#                 ),
#                 dims=[pace.util.Y_DIM, pace.util.X_DIM],
#                 units="m",
#                 origin=(1, 2),
#                 extent=(1, 1),
#             ),
#             (slice(0, 3), slice(None, None)),
#             np.array([[12], [17], [22]]),
#             id="5_by_5_shifted_larger_slice",
#         ),
#     ],
# )
# def test_north(quantity, view_slice, reference):
#     result = quantity.view.north[view_slice]
#     quantity.np.testing.assert_array_equal(result, reference)
#     # result should be a slice of the quantity memory, if it's a slice
#     assert len(result.shape) == 0 or result.base is quantity.data
#     transposed_quantity = pace.util.Quantity(
#         quantity.data.T,
#         dims=quantity.dims[::-1],
#         units=quantity.units,
#         origin=quantity.origin[::-1],
#         extent=quantity.extent[::-1],
#     )
#     transposed_result = transposed_quantity.view.north[view_slice[::-1]]
#     if isinstance(reference, quantity.np.ndarray):
#         quantity.np.testing.assert_array_equal(transposed_result, reference.T)
#     else:
#         quantity.np.testing.assert_array_equal(transposed_result, reference)


@pytest.mark.parametrize(
    "quantity, view_slice, reference",
    [
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (0, 0),
            4,
            id="3_by_3_center_value",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, -1),
            0,
            id="3_by_3_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (slice(-1, 0), slice(-1, 0)),
            np.array([[0]]),
            id="3_by_3_corner_as_slice",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, 0),
            1,
            id="3_by_3_beside_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24],
                    ]
                ),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(2, 2),
                extent=(1, 1),
            ),
            (slice(-2, 0), slice(-1, 2)),
            np.array([[1, 2, 3], [6, 7, 8]]),
            id="5_by_5_larger_slice",
        ),
    ],
)
def test_southwest(quantity, view_slice, reference):
    result = quantity.view.southwest[view_slice]
    quantity.np.testing.assert_array_equal(result, reference)
    # result should be a slice of the quantity memory, if it's a slice
    assert len(result.shape) == 0 or result.base is quantity.data
    transposed_quantity = pace.util.Quantity(
        quantity.data.T,
        dims=quantity.dims[::-1],
        units=quantity.units,
        origin=quantity.origin[::-1],
        extent=quantity.extent[::-1],
    )
    transposed_result = transposed_quantity.view.southwest[view_slice[::-1]]
    if isinstance(reference, quantity.np.ndarray):
        quantity.np.testing.assert_array_equal(transposed_result, reference.T)
    else:
        quantity.np.testing.assert_array_equal(transposed_result, reference)


@pytest.mark.parametrize(
    "quantity, view_slice, reference",
    [
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, 0),
            4,
            id="3_by_3_center_value",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (0, -1),
            6,
            id="3_by_3_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (slice(0, 1), slice(-1, 0)),
            np.array([[6]]),
            id="3_by_3_corner_as_slice",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, -1),
            3,
            id="3_by_3_beside_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24],
                    ]
                ),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(2, 2),
                extent=(1, 1),
            ),
            (slice(-2, 0), slice(-1, 2)),
            np.array([[6, 7, 8], [11, 12, 13]]),
            id="5_by_5_larger_slice",
        ),
    ],
)
def test_southeast(quantity, view_slice, reference):
    result = quantity.view.southeast[view_slice]
    quantity.np.testing.assert_array_equal(result, reference)
    # result should be a slice of the quantity memory, if it's a slice
    assert len(result.shape) == 0 or result.base is quantity.data
    transposed_quantity = pace.util.Quantity(
        quantity.data.T,
        dims=quantity.dims[::-1],
        units=quantity.units,
        origin=quantity.origin[::-1],
        extent=quantity.extent[::-1],
    )
    transposed_result = transposed_quantity.view.southeast[view_slice[::-1]]
    if isinstance(reference, quantity.np.ndarray):
        quantity.np.testing.assert_array_equal(transposed_result, reference.T)
    else:
        quantity.np.testing.assert_array_equal(transposed_result, reference)


@pytest.mark.parametrize(
    "quantity, view_slice, reference",
    [
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.Y_DIM, pace.util.X_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, 0),
            4,
            id="3_by_3_center_value",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.Y_DIM, pace.util.X_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (0, -1),
            6,
            id="3_by_3_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.Y_DIM, pace.util.X_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (slice(0, 1), slice(-1, 0)),
            np.array([[6]]),
            id="3_by_3_corner_as_slice",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.Y_DIM, pace.util.X_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, 0),
            4,
            id="3_by_3_inside_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.Y_DIM, pace.util.X_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, -1),
            3,
            id="3_by_3_beside_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24],
                    ]
                ),
                dims=[pace.util.Y_DIM, pace.util.X_DIM],
                units="m",
                origin=(2, 2),
                extent=(1, 1),
            ),
            (slice(-2, 0), slice(-1, 2)),
            np.array([[6, 7, 8], [11, 12, 13]]),
            id="5_by_5_larger_slice",
        ),
    ],
)
def test_northwest(quantity, view_slice, reference):
    result = quantity.view.northwest[view_slice]
    quantity.np.testing.assert_array_equal(result, reference)
    # result should be a slice of the quantity memory, if it's a slice
    assert len(result.shape) == 0 or result.base is quantity.data
    transposed_quantity = pace.util.Quantity(
        quantity.data.T,
        dims=quantity.dims[::-1],
        units=quantity.units,
        origin=quantity.origin[::-1],
        extent=quantity.extent[::-1],
    )
    transposed_result = transposed_quantity.view.northwest[view_slice[::-1]]
    if isinstance(reference, quantity.np.ndarray):
        quantity.np.testing.assert_array_equal(transposed_result, reference.T)
    else:
        quantity.np.testing.assert_array_equal(transposed_result, reference)


@pytest.mark.parametrize(
    "quantity, view_slice, reference",
    [
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, -1),
            4,
            id="3_by_3_center_value",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (0, 0),
            8,
            id="3_by_3_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (slice(0, 1), slice(0, 1)),
            np.array([[8]]),
            id="3_by_3_corner_as_slice",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, -1),
            4,
            id="3_by_3_inside_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (-1, 0),
            5,
            id="3_by_3_beside_corner",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24],
                    ]
                ),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(2, 2),
                extent=(1, 1),
            ),
            (slice(-2, 0), slice(-1, 2)),
            np.array([[7, 8, 9], [12, 13, 14]]),
            id="5_by_5_larger_slice",
        ),
    ],
)
def test_northeast(quantity, view_slice, reference):
    result = quantity.view.northeast[view_slice]
    quantity.np.testing.assert_array_equal(result, reference)
    # result should be a slice of the quantity memory, if it's a slice
    assert len(result.shape) == 0 or result.base is quantity.data
    transposed_quantity = pace.util.Quantity(
        quantity.data.T,
        dims=quantity.dims[::-1],
        units=quantity.units,
        origin=quantity.origin[::-1],
        extent=quantity.extent[::-1],
    )
    transposed_result = transposed_quantity.view.northeast[view_slice[::-1]]
    if isinstance(reference, quantity.np.ndarray):
        quantity.np.testing.assert_array_equal(transposed_result, reference.T)
    else:
        quantity.np.testing.assert_array_equal(transposed_result, reference)


@pytest.mark.parametrize(
    "quantity, view_slice, reference",
    [
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (0, 0),
            4,
            id="3_by_3_center_value",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (slice(0, 0), slice(0, 0)),
            4,
            id="3_by_3_center_value_as_slice",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(1, 1),
            ),
            (slice(-1, 1), slice(-1, 1)),
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            id="3_by_3_with_halo",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24],
                    ]
                ),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(2, 2),
                extent=(1, 1),
            ),
            (slice(-2, 0), slice(0, 1)),
            np.array([[2, 3], [7, 8], [12, 13]]),
            id="5_by_5_larger_slice",
        ),
        pytest.param(
            pace.util.Quantity(
                np.array(
                    [
                        [0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24],
                    ]
                ),
                dims=[pace.util.X_DIM, pace.util.Y_DIM],
                units="m",
                origin=(1, 1),
                extent=(3, 3),
            ),
            (0,),
            np.array([6, 7, 8]),
            id="5_by_5_one_index",
        ),
    ],
)
def test_interior(quantity, view_slice, reference):
    result = quantity.view.interior[view_slice]
    quantity.np.testing.assert_array_equal(result, reference)
    # result should be a slice of the quantity memory, if it's a slice
    assert len(result.shape) == 0 or result.base is quantity.data
    transposed_quantity = pace.util.Quantity(
        quantity.data.T,
        dims=quantity.dims[::-1],
        units=quantity.units,
        origin=quantity.origin[::-1],
        extent=quantity.extent[::-1],
    )
    if len(view_slice) == len(quantity.dims):  # skip if not
        transposed_result = transposed_quantity.view.interior[view_slice[::-1]]
        if isinstance(reference, quantity.np.ndarray):
            quantity.np.testing.assert_array_equal(transposed_result, reference.T)
        else:
            quantity.np.testing.assert_array_equal(transposed_result, reference)
