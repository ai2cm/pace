#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_namespace_packages, setup


with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

requirements = [
    "f90nml>=1.1.0",
    "gt4py",
    "numpy",
    "pace-util>=0.4.3",
    "xarray",
]

test_requirements = ["pytest==5.2.2", "pytest-subtests>=0.3.0", "serialbox"]

setup(
    author="The Allen Institute for Artificial Intelligence",
    author_email="jeremym@allenai.org",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="fv3core is a gt4py-based FV3 dynamical core for atmospheric models",
    install_requires=requirements,
    extras_require={},
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords="fv3core",
    name="pace-fv3core",
    packages=find_namespace_packages(include=["pace.*"]),
    setup_requires=[],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ai2cm/fv3core",
    version="0.1.0",
    zip_safe=False,
)
