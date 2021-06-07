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

## Build FMS
##---------------------------------------------------------------------------------
FROM fv3gfs-environment AS fv3gfs-fms-install

ENV CC=/usr/bin/mpicc \
    FC=/usr/bin/mpif90 \
    LDFLAGS="-L/usr/lib" \
    LOG_DRIVER_FLAGS="--comments" \
    CPPFLAGS="-I/usr/include -Duse_LARGEFILE -DMAXFIELDMETHODS_=500 -DGFS_PHYS" \
    FCFLAGS="-fcray-pointer -Waliasing -ffree-line-length-none -fno-range-check -fdefault-real-8 -fdefault-double-8 -fopenmp"

COPY FMS /FMS
RUN apt-get update && \
    apt-get install -y m4 libtool autoconf bats && \
    cd /FMS && autoreconf --install && ./configure && \
    cd /FMS && make -j8 && \
    mv /FMS/*/*.mod /FMS/*/*.o /FMS/*/*.h /FMS/

FROM busybox as fv3gfs-fms
COPY --from=fv3gfs-fms-install /FMS /FMS

## Build ESMF
##---------------------------------------------------------------------------------
FROM fv3gfs-environment AS fv3gfs-esmf-install

ENV ESMF_DIR=/esmf \
    ESMF_INSTALL_PREFIX=/usr/local/esmf \
    ESMF_INSTALL_MODDIR=/usr/local/esmf/include \
    ESMF_INSTALL_HEADERDIR=/usr/local/esmf/include \
    ESMF_INSTALL_LIBDIR=/usr/local/esmf/lib \
    ESMF_INSTALL_BINDIR=/usr/local/esmf/bin \
    ESMF_NETCDF_INCLUDE=/usr/include \
    ESMF_NETCDF_LIBS="-lnetcdf -lnetcdff" \
    ESMF_BOPT=O3

RUN git clone -b ESMF_8_0_0 --depth 1 https://git.code.sf.net/p/esmf/esmf $ESMF_DIR && \
    cd $ESMF_DIR && \
    make lib -j24 && \
    make install && \
    make installcheck

FROM busybox as fv3gfs-esmf
COPY --from=fv3gfs-esmf-install /usr/local/esmf $ESMF_DIR

FROM $FMS_IMAGE AS fms_image
FROM $ESMF_IMAGE AS esmf_image


## Build FV3 executable in its own image
##---------------------------------------------------------------------------------
FROM fv3gfs-environment AS fv3gfs-build

ENV FMS_DIR=/FMS \
    ESMF_DIR=/usr/local/esmf

ENV ESMF_INC="-I/usr/local/esmf/include" \
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ESMF_DIR}/lib:${FMS_DIR}/libFMS/.libs/

COPY --from=fms_image /FMS $FMS_DIR
COPY --from=esmf_image /usr/local/esmf $ESMF_DIR

COPY stochastic_physics /stochastic_physics
COPY FV3/coarse_graining /FV3/coarse_graining
COPY FV3/conf /FV3/conf
COPY FV3/ccpp /FV3/ccpp
COPY FV3/cpl /FV3/cpl
COPY FV3/gfsphysics /FV3/gfsphysics
COPY FV3/io /FV3/io
COPY FV3/ipd /FV3/ipd
COPY FV3/stochastic_physics /FV3/stochastic_physics
COPY FV3/makefile FV3/mkDepends.pl FV3/atmos_model.F90 FV3/LICENSE.md \
    FV3/coupler_main.F90 FV3/fv3_cap.F90 FV3/module_fcst_grid_comp.F90 \
    FV3/module_fv3_config.F90 FV3/time_utils.F90 \
    /FV3/

ARG configure_file=configure.fv3.gnu_docker
ARG compile_option

# copy appropriate configuration file to configure.fv3
RUN cp /FV3/conf/$configure_file \
        /FV3/conf/configure.fv3 && \
    if [ ! -z $compile_option ]; then sed -i "33i $compile_option" \
        /FV3/conf/configure.fv3; fi

COPY FV3/atmos_cubed_sphere /FV3/atmos_cubed_sphere

RUN cd /FV3 && make clean_no_dycore && make libs_no_dycore -j16

RUN cd /FV3/atmos_cubed_sphere && make clean && cd /FV3 && make -j16

COPY FV3/write_pkg_config.sh /FV3/
