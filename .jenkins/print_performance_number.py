import json
import os

import numpy as np


for x in os.listdir():
    if x.endswith(".json"):
        f = open(x)
        data = json.load(f)
        for rank in range(6):
            print(
                f"Rank {rank}, mainloop average time: \
                    {np.mean(data['times']['mainloop']['times'][rank][1:])}"
            )
