from typing import List

from setuptools import find_namespace_packages, setup


with open("README.md", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup_requirements: List[str] = []

requirements = [
    "f90nml>=1.1.0",
    "gt4py",
    "numpy",
    "pace-util>=0.4.3",
    "xarray",
]

test_requirements = ["pytest"]

setup(
    author="Vulcan Technologies LLC",
    author_email="elynnw@vulcan.com",
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
    description="pace-physics is a gt4py-based physical parameterization "
    "for atmospheric models",
    install_requires=requirements,
    extras_require={},
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    name="pace-physics",
    packages=find_namespace_packages(include=["pace.*"]),
    setup_requires=[],
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/ai2cm/pace",
    version="0.2.0",
    zip_safe=False,
)
