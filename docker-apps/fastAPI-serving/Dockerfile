# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

ARG directory=./fastAPI-serving

# env in kubernetes einfügen
ENV MLFLOW_TRACKING_URI = "1234"
ENV MLFLOW_MODEL_NAME = "1234"
ENV MLFLOW_MODEL_VERSION = "1234"

COPY /requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt
COPY /app /app


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
