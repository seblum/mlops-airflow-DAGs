FROM python:3.10.11-slim-buster

ARG directory=./cnn-skin-cancer/Docker/preprocessing

COPY $directory/requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt
COPY $directory/app /app

ENTRYPOINT ["python3"]
