FROM tensorflow/tensorflow:2.2.0

RUN pip install tensorflow_data_validation==0.22.0 \
    pip install apache_beam[gcp]==2.20.0 \
    pip install pyarrow==0.16.0 \
    pip install tfx-bsl==0.22.0 \
    pip install ipython==7.15.0

RUN pip download tensorflow_data_validation \
  --no-deps \
  --platform manylinux1_x86_64 \
  --only-binary=:all:

COPY src ./src

ENTRYPOINT ["bash", "./src/data_validation.sh"]


