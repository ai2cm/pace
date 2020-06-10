from setuptools import setup, find_packages

setup_requirements = []

requirements = [
    "xarray>=0.13.0",
    "numpy>=0.15.0",
    "fsspec>=0.6.0",
    "zarr>=2.3.2",
]

test_requirements = []

with open("README.md") as readme_file:
    readme = readme_file.read()


setup(
    author="Vulcan Technologies LLC",
    author_email="jeremym@vulcan.com",
    python_requires=">=3.5",
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
    name="fv3util",
    license="BSD license",
    long_description=readme,
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/VulcanClimateModeling/fv3gfs-python",
    version="0.4.3",
    zip_safe=False,
)
