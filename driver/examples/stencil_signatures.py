import collections
import inspect
from typing import Dict, List, Optional, TextIO, Tuple

import yaml
from gt4py.definitions import FieldInfo

import fv3core.utils.null_comm
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


def report_stencils(
    obj, file: Optional[TextIO]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    sub_objects = {}
    module = inspect.getmodule(obj.__class__)
    print(f"module {module.__name__}, class {obj.__class__.__name__}:", file=file)
    all_access_names: Dict[str, List[str]] = collections.defaultdict(list)
    for name, value in obj.__dict__.items():
        if isinstance(value, pace.dsl.FrozenStencil):
            report_field_info(
                name=name,
                field_info=value.stencil_object.field_info,
                accesses=all_access_names,
                file=file,
            )
        elif has_stencils(value):
            sub_objects[value.__class__] = value
    dependencies = {obj.__class__.__name__: [subobj.__name__ for subobj in sub_objects]}
    for sub_object in sub_objects.values():
        sub_access_names, sub_dependencies = report_stencils(sub_object, file=file)
        dependencies = merge_dicts(dependencies, sub_dependencies)
        for name, values in sub_access_names.items():
            all_access_names[name].extend(values)
    report_aggregate(
        module_name=module.__name__,
        class_name=obj.__class__.__name__,
        accesses=all_access_names,
        file=file,
    )
    return all_access_names, dependencies


def merge_dicts(
    d1: Dict[str, List[str]], d2: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    return_dict = collections.defaultdict(list)
    for key, value in d1.items():
        return_dict[key].extend(value)
    for key, value in d2.items():
        return_dict[key].extend(value)
    return return_dict


def report_field_info(
    name: str,
    field_info: Dict[str, FieldInfo],
    accesses: Dict[str, List[str]],
    file: TextIO,
):
    print(f"    stencil {name}:", file=file)
    for arg_name, arg_field_info in field_info.items():
        if arg_field_info is None:
            access_type = "???"
        else:
            access_type = {"READ": "in", "WRITE": "out", "READ_WRITE": "inout"}[
                arg_field_info.access.name
            ]
        print(f"        {arg_name} ({access_type}):", file=file)
        accesses[arg_name].append(access_type)


def report_aggregate(
    module_name: str,
    class_name: str,
    accesses: Dict[str, List[str]],
    file: TextIO,
):
    print(f"module {module_name}, class {class_name}:", file=file)
    print("    aggregate:", file=file)
    for name, access_names in sorted(accesses.items()):
        print(f"        {name}: {set(access_names)}", file=file)


def report_dependencies(dependencies: Dict[str, List[str]], file: TextIO):
    print(
        """
# this auto generated dotfile maps dependencies between our component classes

digraph {
""",
        file=file,
    )
    for name in dependencies.keys():
        print(f"{name} [shape=box]", file=file)

    print("", file=file)
    for name, dependents in dependencies.items():
        for dependent in sorted(list(set(dependents))):
            print(f"{dependent} -> {name}", file=file)

    print(
        """
}
""",
        file=file,
    )


if __name__ == "__main__":
    with open("configs/baroclinic_c12.yaml", "r") as f:
        driver_config = pace.driver.DriverConfig.from_dict(yaml.safe_load(f))
    driver = pace.driver.Driver(
        config=driver_config,
        comm=fv3core.utils.null_comm.NullComm(rank=0, total_ranks=6),
    )
    with open("stencil_report.txt", "w") as f:
        _, dependencies = report_stencils(driver.dycore, file=f)
    with open("class_dependencies.dot", "w") as f:
        report_dependencies(dependencies, file=f)
