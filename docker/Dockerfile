FROM us.gcr.io/vcm-ml/fv3gfs-environment:serialize

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip
RUN pip3 install pytest==5.2.2 && pip3 install f90nml==1.1.0
COPY gridtools4py gridtools4py 
RUN pip3 install -e /gridtools4py --no-cache-dir
RUN python3 -m gt4py.gt_src_manager install

# FROM us.gcr.io/vcm-ml/dawn-gt4py
# To be able to run unit tests on generated serialized data
COPY --from=us.gcr.io/vcm-ml/fv3gfs-environment:serialize /serialbox2 /serialbox2
COPY fv3 /fv3

