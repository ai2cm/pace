import dataclasses
import unittest.mock

from pace.driver import CreatesComm, CreatesCommSelector, WriterCommConfig


@CreatesCommSelector.register("mock")
@dataclasses.dataclass(frozen=True)
class MockCommConfig(CreatesComm):
    def __post_init__(self):
        self.mock_comm = unittest.mock.MagicMock()
        self.cleaned_up = False

    def get_comm(self):
        assert not self.cleaned_up
        return self.mock_comm

    def cleanup(self):
        self.cleaned_up = True


def test_create_comm_writer():
    config_dict = {
        "type": "write",
        "config": {
            "ranks": [0],
        },
    }
    config = CreatesCommSelector.from_dict(config_dict)
    assert isinstance(config, CreatesCommSelector)
    assert isinstance(config.config, WriterCommConfig)
