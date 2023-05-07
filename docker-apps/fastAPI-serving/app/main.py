from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import os
import pandas as pd
import io
import numpy as np

from PIL import Image
from io import BytesIO


# https://fastapi.tiangolo.com/deployment/docker/#build-a-docker-image-for-fastapi

import mlflow

# Get environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
MLFLOW_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION")


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

# def read_imagefile(data) -> Image.Image:
#     image = Image.open(BytesIO(data))
#     np.array(image)
#     return image

# @app.post("/read")
# async def read_root(file: UploadFile = File(...)):
#     image = read_imagefile(await file.read())
#     return image

# def load_image_into_numpy_array(data):
#     return np.array(Image.open(BytesIO(data)))

# @app.post("/test")
# async def test(file: UploadFile = File(...)):
#     image = load_image_into_numpy_array(await file.read())
#     image_2 = pd.Series(image).to_json(orient='values')
#     return image_2

# @app.post("/upload")
# def upload(file: UploadFile = File(...)):
#     try:
#         contents = file.file.read()
#         with open(file.filename, 'wb') as f:
#             f.write(contents)
#     except Exception:
#         return {"message": "There was an error uploading the file"}
#     finally:
#         file.file.close()

#     return {"message": f"Successfully uploaded {file.filename}"}

def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    np.array(image)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print('[+] Read File')
    image = read_imagefile(await file.read())
    print(image)

    print('[+] Initialize MLflow')
    # Initiate MLflow client
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    print('[+] Load Model')
    model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}")
    print(model)
    
    print('[+] Initiate Prediction')
    #if file.filename.endswith(".json"):
    #return file["media"]
        # image = np.array(file, dtype="uint8")
    data = image / 255
        # os.remove(file.filename)
        
        # # Generate predictions with best model
    preds = model.predict(data)
        
    # Return a JSON object containing the model predictions
    json_compatible_item_data = jsonable_encoder(preds)
    return JSONResponse(content=json_compatible_item_data)
    #else:
        # Raise a HTTP 400 Exception, indicating Bad Request 
        # (you can learn more about HTTP response status codes here)
        #raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")
