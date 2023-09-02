import json
import os
from pathlib import Path

import boto3
import numpy as np
from PIL import Image


def get_image_directory():
    """
    Get the file path for the 'images' directory relative to the current script's location.

    Returns:
        str: The absolute file path to the 'images' directory.
    """
    path = f"{Path(__file__).parent.parent}/images"
    return path


# TODO: insert types
def read_imagefile(data) -> Image.Image:
    image = Image.open(data)
    return image


def preprocess_image(image) -> np.array:
    np_image = np.array(image, dtype="uint8")
    np_image = np_image / 255.0
    np_image = np_image.reshape(1, 224, 224, 3)
    return np_image


def check_status(app_name):
    AWS_REGION = os.getenv("AWS_REGION")
    sage_client = boto3.client("sagemaker", region_name=AWS_REGION)
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description["EndpointStatus"]
    return endpoint_status


def query_endpoint(app_name, data):
    AWS_REGION = os.getenv("AWS_REGION")
    client = boto3.session.Session().client("sagemaker-runtime", AWS_REGION)
    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=data,
        ContentType="application/json",
    )

    prediction = response["Body"].read().decode("ascii")
    prediction = json.loads(prediction)
    print(f"Received response: {prediction}")
    return prediction
