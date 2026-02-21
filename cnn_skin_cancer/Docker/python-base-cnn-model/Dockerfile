FROM python:3.12-slim-bookworm

ARG docker_directory=./cnn_skin_cancer/Docker/python-base-cnn-model
ARG app_directory=./cnn_skin_cancer

WORKDIR /app

# Install Poetry (pinned) and system deps in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "poetry==1.8.5" \
    && poetry config virtualenvs.create false

# Copy lockfiles first for better layer caching
COPY $app_directory/pyproject.toml $app_directory/poetry.lock $app_directory/README.md ./

# Install main dependencies (no dev, no root package)
RUN poetry install --only main --no-root --no-interaction \
    && poetry cache clear pypi --all

# Copy source code and test images
COPY $app_directory/src /app/src
COPY $app_directory/inference_test_images /app/inference_test_images

# Install the root package itself
RUN poetry install --only-root

CMD ["python3"]
