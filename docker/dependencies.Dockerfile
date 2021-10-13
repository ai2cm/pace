ARG serialize=false
ARG BASE_IMAGE
ARG BASE_IMAGE_ENV
ARG MPI_IMAGE

FROM $BASE_IMAGE AS fv3gfs-mpi-install

RUN apt-get update && apt-get install -y \
    wget \
    gcc \
    g++ \
    gfortran \
    make
RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz && \
    tar xzf mpich-3.1.4.tar.gz && \
    cd mpich-3.1.4 && \
    ./configure --enable-fortran --enable-cxx --prefix=/usr --enable-fast=all,O3 && \
    make -j24


FROM busybox as fv3gfs-mpi
COPY --from=fv3gfs-mpi-install /mpich-3.1.4 /mpich-3.1.4

FROM $MPI_IMAGE AS mpi_image

FROM $BASE_IMAGE_ENV AS fv3gfs-environment
ENV DEBIAN_FRONTEND=noninteractive TZ=US/Pacific
RUN apt-get update && apt-get install -y  --no-install-recommends \
    curl \
    wget \
    gcc \
    g++ \
    gfortran \
    git \
    libblas-dev \
    liblapack-dev \
    libtool \
    m4 \
    libnetcdf-dev \
    libnetcdff-dev \
    perl \
    make \
    rsync \
    libffi-dev \
    openssl \
    bats \
    python3 \
    libpython3-dev \
    python3-dev \
    python3-setuptools \
    python3-pip \
    cython3 \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev
ARG PYVERSION=3.8.2
RUN curl https://pyenv.run | bash
ENV PYENV_ROOT /root/.pyenv
ENV PATH="/root/.pyenv/bin:${PATH}"
RUN pyenv update && \
    pyenv install ${PYVERSION} && \
    echo 'eval "$(pyenv init -)"' >> /root/.bashrc && \
    eval "$(pyenv init -)" && \
    pyenv global ${PYVERSION}
ENV PATH="/root/.pyenv/shims:${PATH}"
COPY --from=mpi_image /mpich-3.1.4 /mpich-3.1.4
RUN cd /mpich-3.1.4 && make install && ldconfig && rm -rf /mpich-3.1.4

##
## Setup environment for Serialbox
##---------------------------------------------------------------------------------
FROM fv3gfs-environment as fv3gfs-environment-serialbox-install
# set TZ
ENV DEBIAN_FRONTEND=noninteractive TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install dependencies for serialbox
RUN apt-get update && apt-get install -y \
    libssl-dev \
    clang \
    clang-format \
    clang-tidy \
    python3-numpy \
    python3-nose \
    python3-sphinx

RUN wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz && \
    tar xzf cmake-3.17.3.tar.gz && \
    cd cmake-3.17.3 && \
    ./bootstrap && make -j4 && make install

# Install headers from the Boost library
RUN wget -q https://boostorg.jfrog.io/artifactory/main/release/1.74.0/source/boost_1_74_0.tar.gz && \
    tar xzf boost_1_74_0.tar.gz && \
    cd boost_1_74_0 && \
    cp -r boost /usr/include/ && cd /

###
### Build Serialbox
###
RUN git clone -b v2.6.0 --depth 1 https://github.com/GridTools/serialbox.git /usr/src/serialbox && \
    cmake -B build -S /usr/src/serialbox -DSERIALBOX_USE_NETCDF=ON -DSERIALBOX_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/serialbox && \
    cmake --build build/ -j $(nproc) --target install

FROM busybox as fv3gfs-environment-serialbox
COPY --from=fv3gfs-environment-serialbox-install /usr/local/serialbox /usr/local/serialbox
COPY --from=fv3gfs-environment-serialbox-install /usr/include/boost /usr/include/boost
