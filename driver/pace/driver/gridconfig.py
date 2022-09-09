import dataclasses
from typing import Optional, Union

import numpy as np


@dataclasses.dataclass
class GridConfig:
    stretch_mode: bool = False
    stretch_factor: Optional[np.float64] = None
    lon_target: Optional[np.float_] = None
    lat_target: Optional[np.float_] = None

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
Optional[str] = None