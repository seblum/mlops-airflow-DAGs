FROM python:3.10.11-slim-buster

ARG docker_directory=./cnn_skin_cancer/Docker/model-training
ARG app_directory=./cnn_skin_cancer/src

COPY $docker_directory/requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt

COPY $app_directory/preprocessing.py /app
COPY $app_directory/utils.py /app

CMD ["python3"]
