import os

import yaml

import pace.driver
from pace.driver.run import main


DIR = os.path.dirname(os.path.abspath(__file__))


def test_cpu_orchestration_runs():
    with open(
        os.path.join(
            DIR,
            "../../../driver/examples/configs/baroclinic_c12_orch_cpu.yaml",
        ),
        "r",
    ) as f:
        driver_config = pace.driver.DriverConfig.from_dict(yaml.safe_load(f))
    driver_config.comm_config = pace.driver.NullCommConfig(rank=0, total_ranks=6)
    driver_config.dt_atmos = 60
    driver_config.minutes = 1
    driver_config.hours = 0
    driver_config.days = 0
    driver_config.seconds = 0
    main(driver_config)
