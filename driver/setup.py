from typing import List

from setuptools import find_namespace_packages, setup


setup_requirements: List[str] = []

requirements = [
    "pace-util",
    "fv3core",
    "fv3gfs-physics",
    "pace-stencils",
    "dacite",
    "pyyaml",
    "mpi4py",
    "numpy",
    "netCDF4",
    "xarray",
    "zarr",
]

test_requirements: List[str] = []


setup(
    author="Allen Institute for AI",
    author_email="elynnw@allenai.org",
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
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    name="pace-driver",
    license="BSD license",
    packages=find_namespace_packages(include=["pace.*"]),
    include_package_data=True,
    url="https://github.com/ai2cm/pace",
    version="0.1.0",
    zip_safe=False,
)
