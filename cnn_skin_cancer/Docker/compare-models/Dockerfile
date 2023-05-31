FROM python:3.10.11-slim-buster

ARG docker_directory=./cnn_skin_cancer/Docker/compare-models
ARG app_directory=./cnn_skin_cancer/src

COPY $docker_directory/requirements.txt /app/requirements.txt

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5008

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt
COPY $app_directory/compare_models.py /app

ENTRYPOINT ["python3", "compare_models.py"]
