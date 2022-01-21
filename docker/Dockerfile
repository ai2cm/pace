ARG SERIALBOX_IMAGE
ARG ENVIRONMENT_IMAGE

FROM $SERIALBOX_IMAGE as serialbox_image
FROM $ENVIRONMENT_IMAGE as environment

FROM environment as fv3core

RUN pip3 install --upgrade pip setuptools wheel && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

COPY constraints.txt requirements.txt  /
COPY fv3core/requirements  /fv3core/requirements
COPY fv3gfs-physics/requirements.txt /fv3gfs-physics/requirements.txt
RUN pip3 install -r /requirements.txt -c /constraints.txt

COPY --from=serialbox_image /usr/local/serialbox /usr/local/serialbox
COPY --from=serialbox_image /usr/include/boost /usr/include/boost

###
### Build and install GT4Py
###
ENV BOOST_HOME=/usr/include/boost
ARG CPPFLAGS="-I${BOOST_HOME} -I${BOOST_HOME}/boost"
ARG GT4PY_OPTIONALS=""
COPY external/gt4py /gt4py
#  The following should be removed and change back to gt_src_manager for v37+
RUN git clone --depth 1 -b v2.1.0 https://github.com/GridTools/gridtools.git /gt4py/src/gt4py/_external_src/gridtools2

RUN if [ ! -z `echo $GT4PY_OPTIONALS | grep cuda` ] ; then pip install cupy-cuda102==7.7.0 ; else echo Not installing cuda ; fi
RUN python -m pip install --no-cache-dir -c /constraints.txt -e "/gt4py${GT4PY_OPTIONALS}"

# This environment flag sets rebuild=False in gtscript.stencil calls
ENV FV3_STENCIL_REBUILD_FLAG=False

###
### Copy over necessary source and configuration files
###
COPY pace-util/ pace-util/
COPY fv3core fv3core/
COPY fv3gfs-physics fv3gfs-physics/
COPY stencils stencils/
COPY dsl dsl/
COPY driver driver/
COPY docker/entrypoint.sh /entrypoint.sh

# Docker hard limits shared memory usage. MPICH for oversubscribed situation
# uses shared mem for most of its comunication operations,
# which leads to a sigbus crash.
# Both of those (for version <3.2 and >3.2) will force mpich to go
# through the network stack instead of using the shared nemory
# The cost is a slower runtime
ENV MPIR_CVAR_NOLOCAL=1
ENV MPIR_CVAR_CH3_NOLOCAL=1

RUN chmod +x /entrypoint.sh && \
    /entrypoint.sh

ARG ENV_CUDA_PATH=""
ENV CUDA_PATH=${ENV_CUDA_PATH}
ENV IN_DOCKER=True
