ARG build_image
FROM $build_image as fv3ser
###
### Copy over necessary source and configuration files
###
COPY external/fv3gfs-util/ /usr/src/fv3gfs-util/
COPY fv3core /fv3core/fv3core
COPY tests /fv3core/tests
COPY setup.py setup.cfg HISTORY.md README.md /fv3core/

###
### Install fv3gfs-util
###
RUN pip install --no-cache-dir -e /usr/src/fv3gfs-util

###
### Install fv3core
###
RUN cd /fv3core && \
    rm -rf /fv3core/.git && \
    pip install -e /fv3core
