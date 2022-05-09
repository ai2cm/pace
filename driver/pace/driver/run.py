import dataclasses
import gc
import logging
from typing import Optional

import click
import yaml

from pace.util.mpi import MPI

from .driver import Driver, DriverConfig


logger = logging.getLogger(__name__)


log_levels = {
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def configure_logging(log_rank: Optional[int], log_level: str):
    """
    Configure logging for the driver.

    Args:
        log_rank: rank to log from, or 'all' to log to all ranks,
            forced to 'all' if running without MPI
        log_level: log level to use
    """
    level = log_levels[log_level]
    if MPI is None:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s:%(message)s",
            handlers=[logging.StreamHandler()],
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    else:
        if log_rank is None or int(log_rank) == MPI.COMM_WORLD.Get_rank():
            logging.basicConfig(
                level=level,
                format=(
                    f"%(asctime)s [%(levelname)s] (rank {MPI.COMM_WORLD.Get_rank()}) "
                    "%(name)s:%(message)s"
                ),
                handlers=[logging.StreamHandler()],
                datefmt="%Y-%m-%d %H:%M:%S",
            )


@click.command()
@click.argument(
    "CONFIG_PATH",
    required=True,
    type=click.Path(exists=True, readable=True, dir_okay=False, resolve_path=True),
)
@click.option(
    "--log-rank",
    type=click.INT,
    help="rank to log from, or all ranks by default, ignored if running without MPI",
)
@click.option(
    "--log-level",
    default="info",
    help="one of 'debug', 'info', 'warning', 'error', 'critical'",
)
def command_line(config_path: str, log_rank: Optional[int], log_level: str):
    """
    Run the driver.

    CONFIG_PATH is the path to a DriverConfig yaml file.
    """
    configure_logging(log_rank=log_rank, log_level=log_level)
    logger.info("loading DriverConfig from yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        driver_config = DriverConfig.from_dict(config)
    logging.info(f"DriverConfig loaded: {yaml.dump(dataclasses.asdict(driver_config))}")
    main(driver_config=driver_config)


def main(driver_config: DriverConfig):
    driver = Driver(config=driver_config)
    try:
        driver.step_all()
    finally:
        driver.cleanup()


if __name__ == "__main__":
    command_line()
    # need to cleanup any python objects that may have MPI operations before
    # mpi4py performs its final cleanup
    gc.collect()
