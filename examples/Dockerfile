FROM ubuntu:20.04@sha256:450e066588f42ebe1551f3b1a535034b6aa46cd936fe7f2c6b0d72997ec61dbd

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update -y && \
    apt-get install -y \
    tzdata \
    software-properties-common

RUN add-apt-repository ppa:git-core/ppa -y && \
    apt-get update -y && \
    apt-get install -y \
    git

RUN apt-get update -y && \
    apt-get install -y \
    g++ \
    libgeos-dev \
    libopenmpi-dev \
    libboost-all-dev \
    libhdf5-serial-dev \
    netcdf-bin \
    libnetcdf-dev \
    libproj-dev \
    proj-bin \
    proj-data \
    python3.8 \
    python3.8-dev \
    python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN python3 -m pip install --upgrade pip wheel

COPY . /pace

# 2022.12.22 Added non-editable installs as a workaround for pace module not found, when gt4py was updated to v1.0.
# See https://github.com/ai2cm/pace/issues/419 for more details.
RUN cd /pace && \
    pip3 install -r /pace/requirements_dev.txt -c /pace/constraints.txt && \
    pip install ./driver ./dsl ./fv3core ./physics ./stencils ./util -c /pace/constraints.txt

RUN cd / && \
    git clone https://github.com/ai2cm/fv3net

ENV CFLAGS="-I/usr/include -DACCEPT_USE_OF_DEPRECATED_PROJ_API_H=1"

RUN python3 -m pip install \
    numpy==1.21.2 \
    netCDF4==1.5.7 \
    mpi4py==3.1.0 \
    matplotlib==3.5.2 \
    ipyparallel==8.4.1 \
    jupyterlab==3.4.4 \
    shapely==1.8.5 \
    cartopy==0.18.0 \
    jupyterlab_code_formatter==1.5.2 \
    isort==5.10.1 \
    black==22.3.0 \
    /fv3net/external/vcm

ENV PYTHONPATH=/fv3net/external/fv3viz:/pace/external/gt4py/src

ENV OMPI_ALLOW_RUN_AS_ROOT=1

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

ENV OMPI_MCA_rmaps_base_oversubscribe=1

COPY ./examples/notebooks /notebooks

WORKDIR /notebooks

CMD ["jupyter-lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
