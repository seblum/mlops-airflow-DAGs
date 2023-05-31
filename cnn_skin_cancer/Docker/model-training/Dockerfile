FROM python:3.10.11-slim-buster

ARG docker_directory=./cnn_skin_cancer/Docker/model-training
ARG app_directory=./cnn_skin_cancer/src

COPY $docker_directory/requirements.txt /app/requirements.txt

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5008
ENV MLFLOW_EXPERIMENT_ID=578710797164334239
ENV AWS_BUCKET=testskincancer
ENV MODEL_CLASS=Basic

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt

COPY $app_directory/train.py /app
COPY $app_directory/utils.py /app
COPY $app_directory/model/ /app/model/

ENTRYPOINT ["python3", "train.py"]
