import dataclasses
from typing import Optional, Union

import numpy as np


@dataclasses.dataclass
class GridConfig:
    stretch_mode: Optional[bool] = False
    stretch_factor: Optional[float] = None
    lon_target: Optional[float] = None
    lat_target: Optional[float] = None
    use_tc_vertical_grid: Optional[bool] = None
    tc_ks: Optional[int] = None

    def __post_init__(self):
        if self.stretch_mode:
            if not self.stretch_factor:
                raise ValueError(
                    "Stretch_mode is true, but no stretch_factor is provided."
                )
            if not self.lon_target:
                raise ValueError("Stretch_mode is true, but no lon_target is provided.")
            if not self.lat_target:
                raise ValueError("Stretch_mode is true, but no lat_target is provided.")
