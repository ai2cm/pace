#!/usr/bin/env python3
"""
This script determines whether one of the projects in the repo or any of its
dependencies are different from the version on the `main` branch.

This is useful for running tests only on projects that have changed.

This script should depend only on Python 3.6+ standard libraries.
"""
import argparse
import os
import re
import subprocess
from typing import Any, Dict, Set


DIRNAME = os.path.dirname(os.path.abspath(__file__))
DEPENDENCIES_DOTFILE = os.path.join(DIRNAME, "dependencies.dot")

DEFINITION_PATTERN = re.compile(r"\s*([a-zA-Z0-9]+) \[.*label=\"(.*)\".*\]")
DEPENDENCY_PATTERN = re.compile(r"\s*([a-zA-Z0-9]+) -> ([a-zA-Z0-9]+)")


def get_dependencies() -> Dict[str, Set[str]]:
    name_to_subdir = {}
    subdir_dependencies = {}
    with open(DEPENDENCIES_DOTFILE, "r") as f:
        dotfile_text = f.read()
    for groups in DEFINITION_PATTERN.findall(dotfile_text):
        name_to_subdir[groups[0]] = groups[1]
        subdir_dependencies[groups[1]] = set()
    for groups in DEPENDENCY_PATTERN.findall(dotfile_text):
        name = groups[0]
        dependency_name = groups[1]
        if name in name_to_subdir:
            subdir = name_to_subdir[name]
            dependency_subdir = name_to_subdir[dependency_name]
            subdir_dependencies[subdir].add(dependency_subdir)
    add_nested_dependencies(subdir_dependencies)
    return subdir_dependencies


def add_nested_dependencies(dependency_map: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """
    Given a dictionary mapping keys to dependent keys which may or may
    not contain dependencies of dependencies, update it in-place so that
    dependencies include all sub-dependencies.

    Assumes the dependencies contain no cycles.
    """
    # path can be at most as long as the total number of items
    for _ in range(len(dependency_map)):
        for dependencies in dependency_map.values():
            for dependent_key in dependencies.copy():
                dependencies.update(dependency_map[dependent_key])


def parse_args(subdir_dependencies: Dict[str, Any]):
    parser = argparse.ArgumentParser(
        description=(
            "Determines whether one of the projects in the repo or any of its "
            "dependencies are different from the version on the `main` branch. "
            'Prints "false" if the subdirectory and its dependencies are '
            'unchanged, or "true" if they have changed.'
        )
    )
    parser.add_argument(
        "project_name",
        type=str,
        help="subdirectory name of project to check",
        choices=subdir_dependencies.keys(),
    )
    return parser.parse_args()


def unstaged_files(dirname) -> bool:
    result = subprocess.check_output(
        ["git", "ls-files", "--other", "--directory", "--exclude-standard", dirname]
    )
    return len(result) > 0


def staged_files_changed(dirname) -> bool:
    result = subprocess.check_output(["git", "diff", "main", dirname])
    return len(result) > 0


def changed(dirname) -> bool:
    return unstaged_files(dirname) or staged_files_changed(dirname)


def top_level_files_changed() -> bool:
    exclude_args = [f":!{dirname}/*" for dirname in get_dependencies().keys()]
    result = subprocess.check_output(["git", "diff", "main", "."] + exclude_args)
    return len(result) > 0


if __name__ == "__main__":
    subdir_dependencies = get_dependencies()
    args = parse_args(subdir_dependencies)
    if changed(args.project_name) or top_level_files_changed():
        print("true")
    else:
        for dependency_subdir in subdir_dependencies[args.project_name]:
            if changed(dependency_subdir):
                print("true")
                break
        else:
            print("false")
