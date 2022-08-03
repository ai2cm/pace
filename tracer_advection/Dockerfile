


FROM ubuntu:22.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    python3 python3-dev python3-pip && \
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
    python3-pip && \
    python3 -m pip install --upgrade pip

RUN cd / && \
    git clone https://github.com/ai2cm/fv3net.git && \
    cd fv3net && \
    git checkout

RUN python3 -m pip --no-cache-dir install \
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

COPY notebook /notebook

WORKDIR /notebook

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]