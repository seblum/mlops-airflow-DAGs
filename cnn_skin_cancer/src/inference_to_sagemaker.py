import json
import os
from pathlib import Path

import boto3
import numpy as np
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile


def get_image_directory() -> str:
    """
    Get the file path for the 'inference_test_images' directory relative to the current script's location.

    Returns:
        str: The absolute file path to the 'inference_test_images' directory.
    """
    path = f"{Path(__file__).parent.parent}/inference_test_images"
    return path


def read_imagefile(data: str) -> JpegImageFile:
    """
    Reads an image file and returns it as a PIL JpegImageFile object.

    Args:
        data (str): The file path or binary data representing the image.

    Returns:
        PIL.JpegImagePlugin.JpegImageFile: A PIL JpegImageFile object representing the image.

    Example:
        # Read an image file from a file path
        image_path = "example.jpg"
        image = read_imagefile(image_path)

        # Read an image file from binary data
        with open("example.jpg", "rb") as file:
            binary_data = file.read()
        image = read_imagefile(binary_data)
    """
    image = Image.open(data)
    return image


def preprocess_image(image: JpegImageFile) -> np.array:
    """
    Preprocesses a JPEG image for deep learning models.

    Args:
        image (PIL.JpegImagePlugin.JpegImageFile): A PIL image object in JPEG format.

    Returns:
        np.ndarray: A NumPy array representing the preprocessed image.
                    The image is converted to a NumPy array with data type 'uint8',
                    scaled to values between 0 and 1, and reshaped to (1, 224, 224, 3).

    Example:
        # Load an image using PIL
        image = Image.open("example.jpg")

        # Preprocess the image
        preprocessed_image = preprocess_image(image)
    """
    np_image = np.array(image, dtype="uint8")
    np_image = np_image / 255.0
    np_image = np_image.reshape(1, 224, 224, 3)
    return np_image


def endpoint_status(app_name: str) -> str:
    """
    Checks the status of an Amazon SageMaker endpoint.

    Args:
        app_name (str): The name of the SageMaker endpoint to check.

    Returns:
        str: The current status of the SageMaker endpoint.

    Example:
        # Check the status of a SageMaker endpoint
        endpoint_name = "my-endpoint"
        status = endpoint_status(endpoint_name)
        print(f"Endpoint status: {status}")
    """
    AWS_REGION = os.getenv("AWS_REGION")
    sage_client = boto3.client("sagemaker", region_name=AWS_REGION)
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description["EndpointStatus"]
    return endpoint_status


def query_endpoint(app_name: str, data: str) -> json:
    """
    Queries an Amazon SageMaker endpoint with input data and retrieves predictions.

    Args:
        app_name (str): The name of the SageMaker endpoint to query.
        data (str): Input data in JSON format to send to the endpoint.

    Returns:
        dict: The prediction or response obtained from the SageMaker endpoint.

    Example:
        # Query a SageMaker endpoint with JSON data
        endpoint_name = "my-endpoint"
        input_data = '{"feature1": 0.5, "feature2": 1.2}'
        prediction = query_endpoint(endpoint_name, input_data)
        print(f"Endpoint prediction: {prediction}")
    """
    AWS_REGION = os.getenv("AWS_REGION")
    client = boto3.session.Session().client("sagemaker-runtime", AWS_REGION)
    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=data,
        ContentType="application/json",
    )

    prediction = response["Body"].read().decode("ascii")
    prediction = json.loads(prediction)
    return prediction
