from setuptools import setup, find_packages
import sys

with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_requirements = []

requirements = [
    "f90nml>=1.1.0",
    "gt4py",
    "numpy",
    "fv3gfs-util>=0.4.3",
]

test_requirements = ["serialbox"]

setup(
    author="Vulcan Technologies LLC",
    author_email="elynnw@vulcan.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="fv3gfs-physics is a gt4py-based physical parameterization for atmospheric models",
    install_requires=requirements,
    extras_require={},
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords="fv3gfs-physics",
    name="fv3gfs-physics",
    packages=find_packages(include=["fv3gfs.*"]),
    setup_requires=[],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/VulcanClimateModeling/fv3gfs-physics",
    version="0.1.0",
    zip_safe=False,
)
