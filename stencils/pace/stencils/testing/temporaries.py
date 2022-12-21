import copy
from typing import List

import numpy as np

import pace.util


def copy_temporaries(obj, max_depth: int) -> dict:
    temporaries = {}
    attrs = [a for a in dir(obj) if not a.startswith("__")]
    for attr_name in attrs:
        try:
            attr = getattr(obj, attr_name)
        except AttributeError:
            attr = None
        if isinstance(attr, pace.util.Quantity):
            temporaries[attr_name] = copy.deepcopy(np.asarray(attr.data))
        elif attr.__class__.__module__.split(".")[0] in (  # type: ignore
            "fv3core",
            "pace",
        ):
            if max_depth > 0:
                sub_temporaries = copy_temporaries(attr, max_depth - 1)
                if len(sub_temporaries) > 0:
                    temporaries[attr_name] = sub_temporaries
    return temporaries


def assert_same_temporaries(dict1: dict, dict2: dict):
    diffs = _assert_same_temporaries(dict1, dict2)
    if len(diffs) > 0:
        raise AssertionError(f"{len(diffs)} differing temporaries found: {diffs}")


def _assert_same_temporaries(dict1: dict, dict2: dict) -> List[str]:
    differences = []
    for attr in dict1:
        attr1 = dict1[attr]
        attr2 = dict2[attr]
        if isinstance(attr1, np.ndarray):
            try:
                np.testing.assert_almost_equal(
                    attr1, attr2, err_msg=f"{attr} not equal"
                )
            except AssertionError:
                differences.append(attr)
        else:
            sub_differences = _assert_same_temporaries(attr1, attr2)
            for d in sub_differences:
                differences.append(f"{attr}.{d}")
    return differences
