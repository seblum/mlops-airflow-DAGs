FROM python:3.10.11-slim-buster

COPY $docker_directory/requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt

ENTRYPOINT ["python3"]
