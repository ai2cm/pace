import fv3core


def test_set_backend():
    start_backend = fv3core.get_backend()
    new_backend = "new_backend"
    assert new_backend != start_backend
    fv3core.set_backend(new_backend)
    assert fv3core.get_backend() == new_backend
