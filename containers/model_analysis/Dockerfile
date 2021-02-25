FROM tensorflow/tensorflow:2.2.0

RUN pip install tensorflow_model_analysis==0.22.0 \
    pip install apache_beam[gcp]==2.20.0 \
    pip install pyarrow==0.16.0 \
    pip install tfx-bsl==0.22.0 \
    pip install google-cloud-storage==1.28.0 \
    pip install ipython==7.15.0 

COPY src ./src

ENTRYPOINT ["python", "./src/model_analysis.py"]


