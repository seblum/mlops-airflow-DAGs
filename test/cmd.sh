

docker run -e MLFLOW_TRACKING_URI=http://127.0.0.1:5008/ -e MLFLOW_MODEL_NAME=basic-keras-cnn -e MLFLOW_MODEL_VERSION=2 -p 80:80 fastapi:v1

docker run -e MLFLOW_TRACKING_URI=host.docker.internal:5008/ -e MLFLOW_MODEL_NAME=basic-keras-cnn -e MLFLOW_MODEL_VERSION=2 --network="host" fastapi:v1


docker run -e FASTAPI_SERVING_IP=http://127.0.0.1/ -e FASTAPI_SERVING_PORT=80 -p 8501:8501 streamlit:v1

