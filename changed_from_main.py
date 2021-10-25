"""
This script determines whether one of the projects in the repo or any of its
dependencies are different from the version on the `main` branch.

This is useful for running tests only on projects that have changed.

This script should depend only on Python 3.6+ standard libraries.
"""
import argparse
import re
import os
import subprocess
from typing import Dict, Sequence
import sys


DIRNAME = os.path.dirname(os.path.abspath(__file__))
DEPENDENCIES_DOTFILE = os.path.join(DIRNAME, "dependencies.dot")

DEFINITION_PATTERN = re.compile(r"\s*([a-zA-Z0-9]+) \[.*label=\"(.*)\".*\]")
DEPENDENCY_PATTERN = re.compile(r"\s*([a-zA-Z0-9]+) -> ([a-zA-Z0-9]+)")


def get_dependencies() -> Dict[str, Sequence[str]]:
    name_to_subdir = {}
    subdir_dependencies = {}
    with open(DEPENDENCIES_DOTFILE, "r") as f:
        dotfile_text = f.read()
    for groups in DEFINITION_PATTERN.findall(dotfile_text):
        name_to_subdir[groups[0]] = groups[1]
        subdir_dependencies[groups[1]] = []
    for groups in DEPENDENCY_PATTERN.findall(dotfile_text):
        name = groups[0]
        dependency_name = groups[1]
        if name in name_to_subdir:
            subdir = name_to_subdir[name]
            dependency_subdir = name_to_subdir[dependency_name]
            subdir_dependencies[subdir].append(dependency_subdir)
    return subdir_dependencies

SUBDIR_DEPENDENCIES = get_dependencies()


def parse_args():
    # we use 1 for changed so that if something goes wrong and there is an
    # error, the tests will at least still run
    parser = argparse.ArgumentParser(
        description=(
            "Determines whether one of the projects in the repo or any of its "
            "dependencies are different from the version on the `main` branch. "
            "Prints \"false\" if the subdirectory and its dependencies are "
            "unchanged, or \"true\" if they have changed."
        )
    )
    parser.add_argument(
        "project_name", type=str, help="subdirectory name of project to check", choices=SUBDIR_DEPENDENCIES.keys()
    )
    return parser.parse_args()


def unstaged_files(dirname) -> bool:
    result = subprocess.check_output(
        ["git", "ls-files", "--other", "--directory", "--exclude-standard", dirname]
    )
    return len(result) > 0


def staged_files_changed(dirname) -> bool:
    result = subprocess.check_output(
        ["git", "diff", "main", dirname]
    )
    return len(result) > 0


def changed(dirname) -> bool:
    return unstaged_files(dirname) or staged_files_changed(dirname)


if __name__ == '__main__':
    args = parse_args()
    if changed(args.project_name):
        print("true")
    else:
        for dependency_subdir in SUBDIR_DEPENDENCIES[args.project_name]:
            if changed(dependency_subdir):
                print("true")
                break
        else:
            print("false")
