import os
from typing import Optional

import click

from pace.dsl.dace.utils import count_memory

# Count the memory from a given SDFG
ACTION_SDFG_MEMORY_COUNT = "sdfg_memory_count"


@click.command()
@click.argument(
    "action",
    required=True,
    type=click.Choice([ACTION_SDFG_MEMORY_COUNT]),
    help=f"{ACTION_SDFG_MEMORY_COUNT}: count RAM/VRAM needed for a given SDFG (+/- 10%)",
)
@click.option(
    "--sdfg_path",
    type=click.STRING,
    help="Path to SDFG file",
)
def command_line(action: str, sdfg_path: Optional[str]):
    """
    Run tooling.
    """
    if action == ACTION_SDFG_MEMORY_COUNT:
        if sdfg_path is None or not os.path.exists(sdfg_path):
            raise RuntimeError(f"Can't load SDFG {sdfg_path}")
        count_memory(sdfg_path)


if __name__ == "__main__":
    command_line()
