import io
import os
from io import BytesIO

import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import Image
from tensorflow import keras

# Create FastAPI instance
app = FastAPI()

model_name = "Skin Cancer Detection"
version = "v1.0.0"


@app.get("/info")
async def model_info():
    """
    Endpoint to retrieve information about the model.

    Returns:
        - Dictionary containing the model name and version
    """
    return {"name": model_name, "version": version}


@app.get("/health")
async def service_health():
    """
    Endpoint to check the health status of the service.

    Returns:
        - Dictionary indicating the health status of the service
    """
    return {"ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to make predictions on skin cancer images.

    Parameters:
        - file: Uploaded image file (JPG format)

    Returns:
        - Prediction results as a JSON object
    """
    # Get environment variables
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME")
    MLFLOW_MODEL_VERSION = os.getenv("MLFLOW_MODEL_VERSION")

    # TODO: insert types
    def _read_imagefile(data) -> Image.Image:
        """
        Read image file from bytes data.

        Parameters:
            - data: Bytes data of the image file

        Returns:
            - PIL Image object
        """
        image = Image.open(BytesIO(data))
        return image

    def _preprocess_image(image) -> np.array:
        """
        Preprocess the input image for model prediction.

        Parameters:
            - image: PIL Image object

        Returns:
            - Processed numpy array image
        """
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
