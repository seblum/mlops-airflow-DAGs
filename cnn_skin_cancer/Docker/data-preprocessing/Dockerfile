FROM python:3.10.11-slim-buster

ARG docker_directory=./cnn_skin_cancer/Docker/data-preprocessing
ARG app_directory=./cnn_skin_cancer/src

COPY $docker_directory/requirements.txt /app/requirements.txt

# TODO: give different defaults
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5008
ENV MLFLOW_EXPERIMENT_ID=578710797164334239
ENV AWS_BUCKET=testskincancer
ENV AWS_ACCESS_KEY_ID="AKIA4OKBSADP7ZWFBR46"
ENV AWS_SECRET_ACCESS_KEY="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"


WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt

COPY $app_directory/preprocessing.py /app
COPY $app_directory/utils.py /app

ENTRYPOINT ["python3", "preprocessing.py"]
