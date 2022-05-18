import os
from typing import List

import pytest
import yaml

import pace.driver


dirname = os.path.dirname(os.path.abspath(__file__))

EXAMPLE_CONFIGS_DIR = os.path.join(dirname, "../../examples/configs/")

TESTED_CONFIGS = [
    "baroclinic_c12.yaml",
    "baroclinic_c12_comm_read.yaml",
    "baroclinic_c12_comm_write.yaml",
    "baroclinic_c12_null_comm.yaml",
    "baroclinic_c12_write_restart.yaml",
]

EXCLUDED_CONFIGS: List[str] = []


def test_all_configs_tested_or_excluded():
    """
    If any configs are not tested or excluded, add them to TESTED_CONFIGS or
    EXCLUDED_CONFIGS as appropriate.
    """
    config_files = [
        filename
        for filename in os.listdir(EXAMPLE_CONFIGS_DIR)
        if filename.endswith(".yaml")
    ]
    assert len(config_files) > 0
    missing_files = (
        set(config_files).difference(TESTED_CONFIGS).difference(EXCLUDED_CONFIGS)
    )
    assert len(missing_files) == 0


@pytest.mark.parametrize("filename", TESTED_CONFIGS)
def test_example_config_can_initialize(filename):
    with open(os.path.join(EXAMPLE_CONFIGS_DIR, filename), "r") as f:
        config = pace.driver.DriverConfig.from_dict(yaml.safe_load(f))
    assert isinstance(config, pace.driver.DriverConfig)
