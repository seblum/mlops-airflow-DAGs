FROM python:3.10.11-slim-buster

ARG directory=./streamlit-inference

# env in kubernetes einfügen
ENV FASTAPI_SERVING_IP = "localhost"
ENV FASTAPI_SERVING_PORT = "80"

COPY /requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt
COPY /app /app

#EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
