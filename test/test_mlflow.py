import mlflow

# Initiate MLflow client
MLFLOW_TRACKING_URI = "http://127.0.0.1:5008/"
MLFLOW_MODEL_NAME = "basic-keras-cnn"
MLFLOW_MODEL_VERSION = "2"


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

print("[+] Load Model")
model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}")
print(model)
