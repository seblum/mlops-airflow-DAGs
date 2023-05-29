FROM python:3.10.11-slim-buster

ARG docker_directory=./cnn_skin_cancer/Docker/data-preprocessing
ARG app_directory=./cnn_skin_cancer/src

COPY $docker_directory/requirements.txt /app/requirements.txt

ENV MLFLOW_TRACKING_URI=host.docker.internal:5008
ENV MLFLOW_EXPERIMENT_ID=1234
ENV AWS_BUCKET=aws_bucket

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt

COPY $app_directory/preprocessing.py /app
COPY $app_directory/utils.py /app

ENTRYPOINT ["python3", "preprocessing.py"]
