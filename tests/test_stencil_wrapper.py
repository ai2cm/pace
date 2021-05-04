import contextlib
import unittest.mock

import gt4py.gtscript
import numpy as np
import pytest
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core.decorators
from fv3core.decorators import FrozenStencil, StencilConfig
from fv3core.utils.gt4py_utils import make_storage_from_shape_uncached
from fv3core.utils.typing import FloatField


@contextlib.contextmanager
def mock_gtscript_stencil(mock):
    original_stencil = gt4py.gtscript.stencil
    try:
        gt4py.gtscript.stencil = mock
        yield
    finally:
        gt4py.gtscript.stencil = original_stencil


class MockFieldInfo:
    def __init__(self, axes):
        self.axes = axes


@pytest.mark.parametrize(
    "field_info, origin, field_origins",
    [
        pytest.param(
            {"a": MockFieldInfo(["I"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (1,)},
            id="single_field_I",
        ),
        pytest.param(
            {"a": MockFieldInfo(["J"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (2,)},
            id="single_field_J",
        ),
        pytest.param(
            {"a": MockFieldInfo(["K"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (3,)},
            id="single_field_K",
        ),
        pytest.param(
            {"a": MockFieldInfo(["I", "J"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (1, 2)},
            id="single_field_IJ",
        ),
        pytest.param(
            {"a": MockFieldInfo(["I", "J", "K"])},
            {"_all_": (1, 2, 3), "a": (1, 2, 3)},
            {"_all_": (1, 2, 3), "a": (1, 2, 3)},
            id="single_field_origin_mapping",
        ),
        pytest.param(
            {"a": MockFieldInfo(["I", "J", "K"]), "b": MockFieldInfo(["I"])},
            {"_all_": (1, 2, 3), "a": (1, 2, 3)},
            {"_all_": (1, 2, 3), "a": (1, 2, 3), "b": (1,)},
            id="two_fields_update_origin_mapping",
        ),
        pytest.param(
            {"a": None},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (1, 2, 3)},
            id="single_field_None",
        ),
        pytest.param(
            {"a": MockFieldInfo(["I", "J"]), "b": MockFieldInfo(["I", "J", "K"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (1, 2), "b": (1, 2, 3)},
            id="two_fields",
        ),
    ],
)
def test_compute_field_origins(field_info, origin, field_origins):
    result = fv3core.decorators.compute_field_origins(field_info, origin)
    assert result == field_origins


def copy_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("format_source", [False])
def test_copy_frozen_stencil(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape_uncached((3, 3, 3))
    q_in[:] = 1.0
    q_out = make_storage_from_shape_uncached((3, 3, 3))
    q_out[:] = 2.0
    stencil(q_in, q_out)
    np.testing.assert_array_equal(q_in, q_out)


@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("format_source", [False])
def test_frozen_stencil_raises_if_given_origin(
    backend: str,
    rebuild: bool,
    format_source: bool,
    device_sync: bool,
):
    # only guaranteed when validating args
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=True,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape_uncached((3, 3, 3))
    q_out = make_storage_from_shape_uncached((3, 3, 3))
    with pytest.raises(TypeError, match="origin"):
        stencil(q_in, q_out, origin=(0, 0, 0))


@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("format_source", [False])
def test_frozen_stencil_raises_if_given_domain(
    backend: str,
    rebuild: bool,
    format_source: bool,
    device_sync: bool,
):
    # only guaranteed when validating args
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=True,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape_uncached((3, 3, 3))
    q_out = make_storage_from_shape_uncached((3, 3, 3))
    with pytest.raises(TypeError, match="domain"):
        stencil(q_in, q_out, domain=(3, 3, 3))


@pytest.mark.parametrize(
    "rebuild, validate_args, format_source, device_sync",
    [[False, False, False, False], [True, False, False, False]],
)
def test_frozen_stencil_kwargs_passed_to_init(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil_object = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    ).stencil_object
    mock_stencil = unittest.mock.MagicMock(return_value=stencil_object)
    with mock_gtscript_stencil(mock_stencil):
        FrozenStencil(
            copy_stencil,
            origin=(0, 0, 0),
            domain=(3, 3, 3),
            stencil_config=config,
            externals={},
        )
    mock_stencil.assert_called_once_with(
        definition=copy_stencil, externals={}, **config.stencil_kwargs
    )


def field_after_parameter_stencil(q_in: FloatField, param: float, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = param * q_in


def test_frozen_field_after_parameter(backend):
    config = StencilConfig(
        backend=backend,
        rebuild=False,
        validate_args=False,
        format_source=False,
        device_sync=False,
    )
    result = FrozenStencil(
        field_after_parameter_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("format_source", [False])
def test_copy_non_frozen_stencil(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = fv3core.decorators.gtstencil(
        copy_stencil,
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape_uncached((3, 3, 3))
    q_in[:] = 1.0
    q_out = make_storage_from_shape_uncached((3, 3, 3))
    q_out[:] = 2.0
    stencil(
        q_in,
        q_out,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
    )
    np.testing.assert_array_equal(q_in, q_out)


@pytest.mark.parametrize("validate_args", [False])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [True])
@pytest.mark.parametrize("format_source", [False])
def test_gtscript_with_origin_and_domain_gives_frozen_stencil(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = fv3core.decorators.gtstencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    assert isinstance(stencil, FrozenStencil)


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [False, True])  # even if rebuild=True for gt4py
@pytest.mark.parametrize("format_source", [False])
def test_gtscript_with_origin_and_domain_does_not_rebuild(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = fv3core.decorators.gtstencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    stencil_object = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    ).stencil_object
    mock_stencil = unittest.mock.MagicMock(return_value=stencil_object)
    q_in = make_storage_from_shape_uncached((3, 3, 3))
    q_out = make_storage_from_shape_uncached((3, 3, 3))
    with mock_gtscript_stencil(mock_stencil):
        stencil(
            q_in,
            q_out,
        )
    assert mock_stencil.call_count == 0


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [False, True])  # even if rebuild=True for gt4py
@pytest.mark.parametrize("format_source", [False])
def test_copy_non_frozen_stencil_does_not_rebuild(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = fv3core.decorators.gtstencil(
        copy_stencil,
        stencil_config=config,
        externals={},
    )
    stencil_object = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    ).stencil_object
    mock_stencil = unittest.mock.MagicMock(return_value=stencil_object)
    q_in = make_storage_from_shape_uncached((3, 3, 3))
    q_out = make_storage_from_shape_uncached((3, 3, 3))
    with mock_gtscript_stencil(mock_stencil):
        stencil(
            q_in,
            q_out,
            origin=(0, 0, 0),
            domain=(3, 3, 3),
        )
        mock_stencil.assert_called_once()
        stencil(
            q_in,
            q_out,
            origin=(0, 0, 0),
            domain=(3, 3, 3),
        )
        mock_stencil.assert_called_once()


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [False, True])  # even if rebuild=True for gt4py
@pytest.mark.parametrize("format_source", [False])
def test_copy_non_frozen_stencil_rebuilds_for_new_origin(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = fv3core.decorators.gtstencil(
        copy_stencil,
        stencil_config=config,
        externals={},
    )
    stencil_object = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    ).stencil_object
    mock_stencil = unittest.mock.MagicMock(return_value=stencil_object)
    q_in = make_storage_from_shape_uncached((3, 3, 3))
    q_out = make_storage_from_shape_uncached((3, 3, 3))
    with mock_gtscript_stencil(mock_stencil):
        stencil(
            q_in,
            q_out,
            origin=(0, 0, 0),
            domain=(2, 2, 2),
        )
        stencil(
            q_in,
            q_out,
            origin=(1, 1, 1),
            domain=(2, 2, 2),
        )
    assert mock_stencil.call_count == 2


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [False, True])  # even if rebuild=True for gt4py
@pytest.mark.parametrize("format_source", [False])
def test_copy_non_frozen_stencil_rebuilds_for_new_domain(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = fv3core.decorators.gtstencil(
        copy_stencil,
        stencil_config=config,
        externals={},
    )
    stencil_object = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    ).stencil_object
    mock_stencil = unittest.mock.MagicMock(return_value=stencil_object)
    q_in = make_storage_from_shape_uncached((3, 3, 3))
    q_out = make_storage_from_shape_uncached((3, 3, 3))
    with mock_gtscript_stencil(mock_stencil):
        stencil(
            q_in,
            q_out,
            origin=(0, 0, 0),
            domain=(2, 2, 2),
        )
        stencil(
            q_in,
            q_out,
            origin=(0, 0, 0),
            domain=(3, 3, 3),
        )
    assert mock_stencil.call_count == 2
