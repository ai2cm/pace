from setuptools import setup, find_namespace_packages
import sys

setup_requirements = []

requirements = [
    "cftime>=1.2.1",
    "xarray>=0.15.1",
    "numpy>=0.15.0",
    "fsspec>=0.6.0",
    "zarr>=2.3.2",
]
if sys.version_info.major == 3 and sys.version_info.minor == 6:
    requirements.append("dataclasses")

test_requirements = []

with open("README.md") as readme_file:
    readme = readme_file.read()


with open("HISTORY.md") as history_file:
    history = history_file.read()

setup(
    author="Vulcan Technologies LLC",
    author_email="jeremym@vulcan.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    name="fv3gfs-util",
    license="BSD license",
    long_description=readme + "\n\n" + history,
    packages=find_namespace_packages(include=["fv3gfs.*"]),
    include_package_data=True,
    url="https://github.com/VulcanClimateModeling/fv3gfs-wrapper",
    version="0.5.1",
    zip_safe=False,
)
