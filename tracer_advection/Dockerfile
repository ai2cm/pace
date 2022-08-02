# pace portion
FROM python:3.8.13-bullseye@sha256:2a01d88a1684e6d7f08030cf5ae73b536926c64076cab197e9e3d9f699255283

RUN apt-get update && apt-get install -y make \
    software-properties-common \
    libopenmpi3 \
    libopenmpi-dev \
    libboost-all-dev \
    python3 \
    python3-pip

RUN pip3 install --upgrade setuptools wheel

COPY . /pace

RUN cd /pace && pip3 install -r /pace/requirements_dev.txt -c /pace/constraints.txt

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1



# # jupyter notebook requirements
# RUN cd /pace/tracer_advection

# # get everything ready for cartopy
# RUN apt-get update -y && \
#     apt-get install libgeos-dev && \
#     apt install cmake && \
#     apt install sqlite3 && \ 
#     apt install curl && \
#     apt-get install libcurl4-openssl-dev

# RUN wget https://download.osgeo.org/proj/proj-9.0.0.tar.gz && \
#     tar -xf proj-9.0.0.tar.gz && \
#     cd proj-9.0.0 && \
#     mkdir build && \
#     cd build && \
#     cmake .. && \
#     cmake --build . && \
#     cmake --build . --target install


# # do something about fv3net > external > fv3viz
# RUN pip3 install --no-cache-dir \
#     cartopy==0.20.3 \
#     ipyparallel==8.4.1 \
#     jupyter==1.0.0 \
#     matplotlib==3.5.2\
#     mpi4py==3.1.0a0\
#     netCDF4==1.5.7 \
#     numpy==1.21.2


# Jeremy's portion for running jupyter notebook in Docker
#ARG WRAPPER_IMAGE
#FROM $WRAPPER_IMAGE AS notebook


#RUN ipython profile create --parallel --profile=mpi

#COPY ipcluster_config.py /root/.ipython/profile_mpi/ipcluster_config.py

#WORKDIR .

#CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]