# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

ARG directory=./streamlit-inferece

# env in kubernetes einfügen
ENV FASTAPI_URL = "localhost"
ENV FASTAPI_PORT = "8000"

COPY $directory/requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade -r requirements.txt
COPY $directory/app /app

#EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["streamlit.py"]