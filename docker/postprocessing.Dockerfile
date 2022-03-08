FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt install -y --no-install-recommends \
    software-properties-common

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends\
    g++ \
    gcc \
    gfortran \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev

RUN apt-get update -y && \
    apt install -y --no-install-recommends \
    git \
    python3.9 \
    python3.9-dev &&\
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 60

RUN apt-get update -y &&\
    apt install -y --no-install-recommends\
    python3-pip

RUN python -m pip --no-cache-dir install --upgrade pip && \
    python -m pip --no-cache-dir install setuptools &&\
    python -m pip --no-cache-dir install wheel

RUN python -m pip --no-cache-dir \
    install \
    numpy \
    matplotlib \
    cython \
    cartopy \
    xarray \
    zarr

# default shapely causes seg fault
RUN pip uninstall shapely -y
RUN pip install --no-binary :all: shapely

# set up for fv3viz
RUN cd /
RUN git clone https://github.com/ai2cm/fv3net.git
RUN cd fv3net && git checkout 9df1bde7
RUN python -m pip install fv3net/external/vcm
ENV PYTHONPATH=/fv3net/external/fv3viz
