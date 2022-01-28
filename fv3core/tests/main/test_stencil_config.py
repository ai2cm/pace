import pytest

from pace.dsl.stencil import StencilConfig


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("device_sync", [True, False])
@pytest.mark.parametrize("rebuild", [True, False])
@pytest.mark.parametrize("format_source", [True, False])
def test_same_config_equal(
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
    assert config == config

    same_config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    assert config == same_config


@pytest.mark.parametrize("validate_args", [True])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [True])
@pytest.mark.parametrize("format_source", [True])
def test_different_backend_not_equal(
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

    different_config = StencilConfig(
        backend="fakebackend",
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    assert config != different_config


@pytest.mark.parametrize("validate_args", [True])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [True])
@pytest.mark.parametrize("format_source", [True])
def test_different_rebuild_not_equal(
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

    different_config = StencilConfig(
        backend=backend,
        rebuild=not rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    assert config != different_config


@pytest.mark.parametrize("validate_args", [True])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [True])
@pytest.mark.parametrize("format_source", [True])
def test_different_device_sync_not_equal(
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

    different_config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=not device_sync,
    )
    assert config != different_config


@pytest.mark.parametrize("validate_args", [True])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [True])
@pytest.mark.parametrize("format_source", [True])
def test_different_validate_args_not_equal(
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

    different_config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=not validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    assert config != different_config


@pytest.mark.parametrize("validate_args", [True])
@pytest.mark.parametrize("device_sync", [True])
@pytest.mark.parametrize("rebuild", [True])
@pytest.mark.parametrize("format_source", [True])
def test_different_format_source_not_equal(
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

    different_config = StencilConfig(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=not format_source,
        device_sync=device_sync,
    )
    assert config != different_config
