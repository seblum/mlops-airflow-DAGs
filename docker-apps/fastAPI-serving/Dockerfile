# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

ARG directory=./fastAPI-serving

# env in kubernetes einfügen
ENV MLFLOW_EXPERIMENT_ID = "1234"
ENV MLFLOW_RUN_ID = "4321"

COPY $directory/requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade -r requirements.txt
COPY $directory/app /app


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]