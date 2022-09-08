import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class GridConfig:
    stretch_mode: bool = False
    stretch_factor: Optional[np.float_] = 1.0
    lon_target: Optional[np.float_] = 0.0
    lat_target: Optional[np.float_] = 0.0

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
