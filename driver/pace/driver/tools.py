import os
from typing import Optional

import click

from pace.dsl.dace.utils import count_memory_from_path


# Count the memory from a given SDFG
ACTION_SDFG_MEMORY_COUNT = "sdfg_memory_count"


@click.command()
@click.argument(
    "action",
    required=True,
    type=click.Choice([ACTION_SDFG_MEMORY_COUNT]),
)
@click.option(
    "--sdfg_path",
    type=click.STRING,
)
@click.option("--report_detail", is_flag=True, type=click.BOOL, default=False)
def command_line(action: str, sdfg_path: Optional[str], report_detail: Optional[bool]):
    """
    Run tooling.
    """
    if action == ACTION_SDFG_MEMORY_COUNT:
        if sdfg_path is None or not os.path.exists(sdfg_path):
            raise RuntimeError(f"Can't load SDFG {sdfg_path}")
        print(count_memory_from_path(sdfg_path, detail_report=report_detail))


if __name__ == "__main__":
    command_line()
