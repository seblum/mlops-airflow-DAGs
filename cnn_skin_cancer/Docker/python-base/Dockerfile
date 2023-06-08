FROM python:3.10.11-slim-buster

ARG docker_directory=./cnn_skin_cancer/Docker/python-base
ARG app_directory=./cnn_skin_cancer/src

COPY $docker_directory/requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade -r requirements.txt

RUN pip3 install poetry && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction && \
    poetry cache clear pypi --all

COPY $app_directory/ /app
RUN poetry install --only-root


CMD ["python3"]
