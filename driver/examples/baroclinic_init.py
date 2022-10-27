from argparse import ArgumentParser

import yaml

from pace.driver.run import Driver, DriverConfig


def parse_args():
    usage = "usage: python %(prog)s config_file"
    parser = ArgumentParser(usage=usage)

    parser.add_argument(
        "config_file",
        type=str,
        action="store",
        help="which config file to use",
    )
    return parser.parse_args()


args = parse_args()
with open(args.config_file, "r") as f:
    driver_config = DriverConfig.from_dict(yaml.safe_load(f))
driver = Driver(config=driver_config)
driver.diagnostics.store(time=driver.config.start_time, state=driver.state)
driver.diagnostics.store_grid(grid_data=driver.state.grid_data)
