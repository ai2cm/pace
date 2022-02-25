FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&\
    apt install -y --no-install-recommends \
    software-properties-common

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends\
    g++ \
    gcc \
    gfortran \
    make \
    wget \
    pkg-config \
    libsqlite3-dev \
    sqlite3 \
    libcurl4-openssl-dev \
    libtiff5 \
    libtiff5-dev \
    openssl \
    libssl-dev

RUN cd / && \
    wget https://github.com/Kitware/CMake/releases/download/v3.22.2/cmake-3.22.2.tar.gz && \
    tar xzf cmake-3.22.2.tar.gz && \
    rm cmake-3.22.2.tar.gz && \
    cd cmake-3.22.2 && \
    ./bootstrap && make -j4 && make install

RUN cd / && \
    wget http://download.osgeo.org/geos/geos-3.8.2.tar.bz2 && \
    tar xfj geos-3.8.2.tar.bz2 && \
    rm geos-3.8.2.tar.bz2 && \
    cd geos-3.8.2 && \
    mkdir _build && \
    cd _build && \
    cmake .. && make -j4 && ctest && make install

RUN cd / && \
    wget http://archive.ubuntu.com/ubuntu/pool/universe/p/proj/proj_8.2.1.orig.tar.gz && \
    tar xzf proj_8.2.1.orig.tar.gz && \
    rm proj_8.2.1.orig.tar.gz && \
    cd proj-8.2.1 && \
    ./configure && make -j4 && make install

RUN apt-get update -y && \
    apt install -y --no-install-recommends \
    git \
    python3.8 \
    python3.8-dev &&\
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 60

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

# set up for fv3viz
RUN cd /
RUN git clone https://github.com/ai2cm/fv3net.git
RUN cd fv3net && git checkout 9df1bde7
RUN python -m pip install fv3net/external/vcm
