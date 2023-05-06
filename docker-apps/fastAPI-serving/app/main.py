from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import File
import os
import pandas as pd
import io

# https://fastapi.tiangolo.com/deployment/docker/#build-a-docker-image-for-fastapi

import mlflow

# Get environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
MLFLOW_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION")
MLFLOW_EXPERIMENT_ID = os.getenv("MLFLOW_EXPERIMENT_ID")
MLFLOW_RUN_ID = os.getenv("MLFLOW_RUN_ID")


# Create FastAPI instance
app = FastAPI()

model_name = "Skin Cancer Detection"
version = "v1.0.0"

@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post("/predict")
async def predict(file: bytes = File(...)):
    print('[+] Initialize MLflow')

    # Initiate MLflow client
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Load best model (based on logloss) amongst all runs in all experiments
    # all_exps = [exp.experiment_id for exp in client.list_experiments()]
    # runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
    # run_id, exp_id = runs.loc[runs['metrics.log_loss'].idxmin()]['run_id'], runs.loc[runs['metrics.log_loss'].idxmin()]['experiment_id']
    print('[+] Load Model')
    best_model = mlflow.pyfunc.load_model(f"mlruns/{MLFLOW_EXPERIMENT_ID}/{MLFLOW_RUN_ID}/artifacts/model/")
    # model_uri

    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    # test_h2o = h2o.H2OFrame(test_df)

    # Generate predictions with best model
    preds = best_model.predict(test_df)
    
    json_compatible_item_data = jsonable_encoder(preds)
    return JSONResponse(content=json_compatible_item_data)