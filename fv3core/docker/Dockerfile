ARG SERIALBOX_IMAGE
ARG ENVIRONMENT_IMAGE

FROM $SERIALBOX_IMAGE as serialbox_image
FROM $ENVIRONMENT_IMAGE as environment

FROM environment as fv3core

RUN pip3 install --upgrade pip setuptools wheel && \
    ln -s /bin/python3 /bin/python && \
    ln -s /bin/pip3 /bin/pip

COPY constraints.txt requirements.txt  /
COPY requirements  /requirements
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

RUN if [ ! -z `echo $GT4PY_OPTIONALS | grep cuda` ] ; then pip install cupy-cuda102==7.7.0 ; else echo Not installing cuda ; fi
RUN pip install --no-cache-dir -c /constraints.txt "/gt4py${GT4PY_OPTIONALS}" && \
    python3 -m gt4py.gt_src_manager install --major-version 2

# This environment flag sets rebuild=False in gtscript.stencil calls
ENV FV3_STENCIL_REBUILD_FLAG=False

###
### Copy over necessary source and configuration files
###
COPY external/pace-util/ /external/pace-util/
COPY external/stencils/ /external/stencils/
COPY external/dsl/ /external/dsl/
COPY fv3core /fv3core/fv3core
COPY tests /fv3core/tests
COPY setup.py README.md /fv3core/
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
