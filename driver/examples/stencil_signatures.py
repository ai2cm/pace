import collections
import inspect
from typing import Optional, TextIO

import yaml

import pace.driver
import pace.dsl
import pace.util


def has_stencils(object):
    for name in dir(object):
        try:
            stencil_found = isinstance(getattr(object, name), pace.dsl.FrozenStencil)
        except (AttributeError, RuntimeError):
            stencil_found = False
        if stencil_found:
            return True
    return False


def report_stencils(obj, file: Optional[TextIO]):
    sub_objects = {}
    module = inspect.getmodule(obj.__class__)
    print(f"module {module.__name__}, class {obj.__class__.__name__}:", file=file)
    all_access_names = collections.defaultdict(list)
    for name, value in obj.__dict__.items():
        if isinstance(value, pace.dsl.FrozenStencil):
            print(f"    stencil {name}:", file=file)
            for arg_name, field_info in value.stencil_object.field_info.items():
                if field_info is None:
                    access_type = "???"
                else:
                    access_type = {"READ": "in", "WRITE": "out", "READ_WRITE": "inout"}[
                        field_info.access.name
                    ]
                print(f"        {arg_name} ({access_type}):", file=file)
                all_access_names[arg_name].append(access_type)
        elif has_stencils(value):
            sub_objects[value.__class__] = value
    for sub_object in sub_objects.values():
        sub_access_names = report_stencils(sub_object, file=file)
        for name, values in sub_access_names.items():
            all_access_names[name].extend(values)
    print(f"module {module.__name__}, class {obj.__class__.__name__}:", file=file)
    print("    aggregate:", file=file)
    for name, access_names in sorted(all_access_names.items()):
        print(f"        {name}: {set(access_names)}", file=file)
    return all_access_names


if __name__ == "__main__":
    with open("configs/baroclinic_c12.yaml", "r") as f:
        driver_config = pace.driver.DriverConfig.from_dict(yaml.safe_load(f))
    driver_config.comm_config = pace.driver.CreatesCommSelector(
        config=pace.driver.NullCommConfig(rank=0, total_ranks=6), type="null"
    )
    driver = pace.driver.Driver(config=driver_config)
    with open("stencil_report.txt", "w") as f:
        report_stencils(driver.dycore, file=f)
