from typing import List

from setuptools import find_namespace_packages, setup


setup_requirements: List[str] = []

requirements = [
    "cftime>=1.2.1",
    "numpy>=0.15.0",
    "fsspec>=0.6.0",
    "typing_extensions>=3.7.4",
    "f90nml>=1.1.0",
]

test_requirements: List[str] = []

with open("README.md") as readme_file:
    readme = readme_file.read()


with open("HISTORY.md") as history_file:
    history = history_file.read()

setup(
    author="Allen Institute of Artificial Intelligence",
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
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    extras_require={
        "netcdf": ["xarray>=0.15.1", "scipy>=1.3.1"],
        "zarr": ["zarr>=2.3.2", "xarray>=0.15.1", "scipy>=1.3.1"],
    },
    name="pace-util",
    license="BSD license",
    long_description=readme + "\n\n" + history,
    packages=find_namespace_packages(include=["pace.*"]),
    include_package_data=True,
    url="https://github.com/ai2cm/pace",
    version="0.9.0",
    zip_safe=False,
)
