ARG serialize=false
ARG BASE_IMAGE
ARG BASE_IMAGE_ENV
ARG MPI_IMAGE
ARG FMS_IMAGE
ARG ESMF_IMAGE

FROM $BASE_IMAGE AS fv3gfs-mpi-install

RUN apt-get update && apt-get install -y \
    wget \
    gcc \
    g++ \
    gfortran \
    make
#RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 && \
#    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8 && \
#    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-8 8
RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz && \
    tar xzf mpich-3.1.4.tar.gz && \
    cd mpich-3.1.4 && \
    ./configure --enable-fortran --enable-cxx --prefix=/usr --enable-fast=all,O3 && \
    make -j24


FROM busybox as fv3gfs-mpi
COPY --from=fv3gfs-mpi-install /mpich-3.1.4 /mpich-3.1.4

FROM $MPI_IMAGE AS mpi_image

FROM $BASE_IMAGE_ENV AS fv3core-environment
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
#RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 && \
#    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8 && \
#    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-8 8
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
FROM fv3core-environment as fv3core-environment-serialbox-install
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
COPY --from=fv3core-environment-serialbox-install /usr/local/serialbox /usr/local/serialbox
COPY --from=fv3core-environment-serialbox-install /usr/include/boost /usr/include/boost

###
### Set Fortran environment
###
FROM $BASE_IMAGE_ENV AS fv3gfs-environment

RUN apt-get update && apt-get install -y \
    wget \
    gcc \
    g++ \
    gfortran \
    make \
    curl \
    git \
    libblas-dev \
    liblapack-dev \
    libnetcdf-dev \
    libnetcdff-dev \
    perl \
    rsync \
    libffi-dev \
    openssl
#RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 && \
#    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8 && \
#    update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-8 8
COPY --from=mpi_image /mpich-3.1.4 /mpich-3.1.4
RUN cd /mpich-3.1.4 && make install && ldconfig

# download and install NCEP libraries
RUN git config --global http.sslverify false && \
    git clone https://github.com/NCAR/NCEPlibs.git && \
    mkdir /opt/NCEPlibs && \
    cd NCEPlibs && \
    git checkout 3da51e139d5cd731c9fc27f39d88cb4e1328212b && \
    echo "y" | ./make_ncep_libs.sh -s linux -c gnu -d /opt/NCEPlibs -o 1
