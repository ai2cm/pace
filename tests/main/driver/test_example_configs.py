import os
from typing import List

import pytest
import yaml

import pace.driver


dirname = os.path.dirname(os.path.abspath(__file__))

EXAMPLE_CONFIGS_DIR = os.path.join(dirname, "../../../driver/examples/configs/")

TESTED_CONFIGS: List[str] = [
    "baroclinic_c12.yaml",
    "baroclinic_c12_comm_read.yaml",
    "baroclinic_c12_comm_write.yaml",
    "baroclinic_c12_null_comm.yaml",
    "baroclinic_c12_write_restart.yaml",
    "baroclinic_c48_6ranks_serialbox_test.yaml",
]
EXCLUDED_CONFIGS: List[str] = [
    # We don't test serialbox example because it loads namelist
    # filepath that are not in git
    "baroclinic_c12_from_serialbox.yaml",
    "baroclinic_c12_orch_cpu.yaml",
    "tropical_read_restart_fortran.yml",
    "tropicalcyclone_c128.yaml",
]

JENKINS_CONFIGS_DIR = os.path.join(dirname, "../../../.jenkins/driver_configs/")
TESTED_JENKINS_CONFIGS: List[str] = [
    "baroclinic_c48_6ranks_dycore_only.yaml",
    "baroclinic_c192_6ranks.yaml",
    "baroclinic_c192_54ranks.yaml",
]

EXCLUDED_JENKINS_CONFIGS: List[str] = [
    # We don't test serialbox example because it loads namelist
    # filepath that are not in git
    "baroclinic_c48_6ranks_dycore_only_serialbox.yaml",
]


@pytest.mark.parametrize(
    "config_dir, tested_configs, excluded_configs",
    [
        pytest.param(
            EXAMPLE_CONFIGS_DIR, TESTED_CONFIGS, EXCLUDED_CONFIGS, id="example configs"
        ),
        pytest.param(
            JENKINS_CONFIGS_DIR,
            TESTED_JENKINS_CONFIGS,
            EXCLUDED_JENKINS_CONFIGS,
            id="jenkins configs",
        ),
    ],
)
def test_all_configs_tested_or_excluded(
    config_dir: str, tested_configs: List[str], excluded_configs: List[str]
):
    """
    If any configs are not tested or excluded, add them to TESTED_CONFIGS or
    EXCLUDED_CONFIGS as appropriate.
    """
    config_files = [
        filename for filename in os.listdir(config_dir) if filename.endswith(".yaml")
    ]
    assert len(config_files) > 0
    missing_files = (
        set(config_files).difference(tested_configs).difference(excluded_configs)
    )
    assert len(missing_files) == 0


@pytest.mark.parametrize(
    "path, file_list",
    [
        pytest.param(EXAMPLE_CONFIGS_DIR, TESTED_CONFIGS),
        pytest.param(JENKINS_CONFIGS_DIR, TESTED_JENKINS_CONFIGS),
    ],
)
def test_example_config_can_initialize(path: str, file_list: List[str]):
    for file_name in file_list:
        with open(os.path.join(path, file_name), "r") as f:
            config = pace.driver.DriverConfig.from_dict(yaml.safe_load(f))
        assert isinstance(config, pace.driver.DriverConfig)
