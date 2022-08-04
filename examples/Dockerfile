
#FROM python:3.8.13-bullseye@sha256:2a01d88a1684e6d7f08030cf5ae73b536926c64076cab197e9e3d9f699255283
FROM ubuntu:22.04


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
    libproj-dev \
    proj-bin \
    proj-data \
    python3 python3-dev python3-pip && \
    python3 -m pip install --upgrade pip wheel

COPY . /pace

RUN cd /pace && pip3 install -r /pace/requirements_dev.txt -c /pace/constraints.txt


RUN cd / && \
    git clone https://github.com/ai2cm/fv3net.git && \
    cd fv3net && \
    git checkout

RUN python3 -m pip install \
    cartopy \
    fv3net/external/vcm \
    ipyparallel==8.4.1 \
    jupyter==1.0.0 \
    matplotlib==3.5.2 \
    mpi4py==3.1.0 \
    netCDF4 \
    numpy

ENV PYTHONPATH=/fv3net/external/fv3viz

ENV OMPI_ALLOW_RUN_AS_ROOT=1

ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

COPY tracer_advection/notebook /notebook

WORKDIR /notebook

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
