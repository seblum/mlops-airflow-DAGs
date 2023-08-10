FROM python:3.10.11-slim-buster

ARG docker_directory=./cnn_skin_cancer/Docker/fastapi-serve-app

ENV MLFLOW_TRACKING_URI=host.docker.internal:5008
ENV MLFLOW_MODEL_NAME=basic-keras-cnn
ENV MLFLOW_MODEL_VERSION=5

COPY $docker_directory/requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt
COPY $docker_directory/app /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
