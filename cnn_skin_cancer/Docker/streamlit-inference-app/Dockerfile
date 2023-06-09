FROM python:3.10.11-slim-buster

ARG docker_directory=./cnn_skin_cancer/Docker/streamlit-inference-app

ENV FASTAPI_SERVING_IP=host.docker.internal
ENV FASTAPI_SERVING_PORT=80

COPY $docker_directory/requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt
COPY $docker_directory/app /app

#EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
