FROM python:3.8.13-bullseye@sha256:2a01d88a1684e6d7f08030cf5ae73b536926c64076cab197e9e3d9f699255283

RUN apt-get update && apt-get install -y make \
    software-properties-common \
    libopenmpi3 \
    libopenmpi-dev \
    libboost-all-dev \
    libhdf5-serial-dev \
    netcdf-bin \
    libnetcdf-dev \
    python3 \
    python3-pip

RUN pip3 install --upgrade setuptools wheel

COPY constraints.txt /pace/constraints.txt

RUN pip3 install -r /pace/constraints.txt

COPY . /pace

RUN cd /pace && \
    pip3 install -r /pace/requirements_dev.txt -c /pace/constraints.txt && \
    python3 -m gt4py.gt_src_manager install

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
