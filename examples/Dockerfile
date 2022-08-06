#FROM python:3.8.13-bullseye@sha256:2a01d88a1684e6d7f08030cf5ae73b536926c64076cab197e9e3d9f699255283
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update -y && \
    apt-get install -y tzdata

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    git && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends\
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
    python3.8 python3.8-dev python3-pip && \
    python3 -m pip install --upgrade pip wheel

COPY . /pace

RUN cd /pace && pip3 install -r /pace/requirements_dev.txt -c /pace/constraints.txt

RUN cd / && \
    git clone https://github.com/ai2cm/fv3net.git && \
    cd fv3net && \
    git checkout

ENV CFLAGS="-I/usr/include -DACCEPT_USE_OF_DEPRECATED_PROJ_API_H=1"

RUN python3 -m pip install \
    cartopy==0.18.0 \
    fv3net/external/vcm \
    ipyparallel==8.4.1 \
    jupyter==1.0.0 \
    matplotlib==3.5.2 \
    mpi4py==3.1.0 \
    netCDF4 \
    numpy 

ENV PYTHONPATH=/fv3net/external/fv3viz:/pace/external/gt4py/src

ENV OMPI_ALLOW_RUN_AS_ROOT=1

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

COPY ./examples/notebooks /notebooks

WORKDIR /notebooks

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
