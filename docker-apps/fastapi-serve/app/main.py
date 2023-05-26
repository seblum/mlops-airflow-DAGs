from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from tensorflow import keras
from PIL import Image
from io import BytesIO

import os
import pandas as pd
import io
import numpy as np
import mlflow
import mlflow.keras


# Get environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
MLFLOW_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION")


# Create FastAPI instance
app = FastAPI()

model_name = "Skin Cancer Detection"
version = "v1.0.0"


@app.get("/info")
async def model_info():
    """Return model information, version, how to call"""
    return {"name": model_name, "version": version}


@app.get("/health")
async def service_health():
    """Return service health"""
    return {"ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # TODO: insert types
    def _read_imagefile(data) -> Image.Image:
        image = Image.open(BytesIO(data))
        return image

    def _preprocess_image(image) -> np.array:
        np_image = np.array(image, dtype="uint8")
        np_image = np_image / 255.0
        np_image = np_image.reshape(1, 224, 224, 3)
        return np_image
    
    if file.filename.endswith(".jpg"):
        print("[+] Read File")
        image = _read_imagefile(await file.read())

        print("[+] Initialize MLflow")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        print("[+] Load Model")
        model = mlflow.keras.load_model(f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_VERSION}")

        print("[+] Preprocess Data")
        np_image = _preprocess_image(image)

        print("[+] Initiate Prediction")
        preds = model.predict(np_image)

        print("[+] Return Model Prediction")
        return {"prediction": preds.tolist()}
    else:
        # Raise a HTTP 400 Exception, indicating Bad Request
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG Files accepted.")
