.. highlight:: shell

============
Installation
============

Shell scripts to install Pace on specific machines such as Gaea can be found in `examples/build_scripts/`.

When cloning Pace you will need to update the repository's submodules as well:

.. code-block:: console

    $ git clone --recursive https://github.com/ai2cm/pace.git

or if you have already cloned the repository:

.. code-block:: console

    $ git submodule update --init --recursive


Pace requires GCC > 9.2, MPI, and Python 3.8 on your system, and CUDA is required to run with a GPU backend.
You will also need the headers of the boost libraries in your `$PATH` (boost itself does not need to be installed).

.. code-block:: console

    $ cd BOOST/ROOT
    $ wget https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
    $ tar -xzf boost_1_79_0.tar.gz
    $ mkdir -p boost_1_79_0/include
    $ mv boost_1_79_0/boost boost_1_79_0/include/
    $ export BOOST_ROOT=BOOST/ROOT/boost_1_79_0


We recommend creating a python `venv` or conda environment specifically for Pace.

.. code-block:: console

    $ python3 -m venv venv_name
    $ source venv_name/bin/activate

Inside of your pace `venv` or conda environment pip install the Python requirements, GT4Py, and Pace:

.. code-block:: console

    $ pip3 install -r requirements_dev.txt -c constraints.txt

There are also separate requirements files which can be installed for linting (`requirements_lint.txt`) and building documentation (`requirements_docs.txt`).
