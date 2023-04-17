from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

# https://fastapi.tiangolo.com/deployment/docker/#build-a-docker-image-for-fastapi

import mlflow
# Create FastAPI instance
app = FastAPI()

# Initiate H2O instance and MLflow client
#h2o.init()
client = MlflowClient()

# Load best model (based on logloss) amongst all runs in all experiments
# all_exps = [exp.experiment_id for exp in client.list_experiments()]
# runs = mlflow.search_runs(experiment_ids=all_exps, run_view_type=ViewType.ALL)
# run_id, exp_id = runs.loc[runs['metrics.log_loss'].idxmin()]['run_id'], runs.loc[runs['metrics.log_loss'].idxmin()]['experiment_id']
best_model = mlflow.h2o.load_model(f"mlruns/{exp_id}/{run_id}/artifacts/model/")
# model_uri

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
    print('[+] Initiate Prediction')
    file_obj = io.BytesIO(file)
    test_df = pd.read_csv(file_obj)
    test_h2o = h2o.H2OFrame(test_df)

    # Generate predictions with best model (output is H2O frame)
    preds = best_model.predict(X_h2o)
    
    json_compatible_item_data = jsonable_encoder(preds)
    return JSONResponse(content=json_compatible_item_data)