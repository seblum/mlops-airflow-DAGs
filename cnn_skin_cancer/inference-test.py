import json

import boto3
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

global app_name
global region

endpoint_name = "test-cnn-skin-cancer"

app_name = endpoint_name
region = "eu-central-1"


from io import BytesIO

import numpy as np

#####
from PIL import Image


# TODO: insert types
def _read_imagefile(data) -> Image.Image:
    image = Image.open(data)
    return image


def _preprocess_image(image) -> np.array:
    np_image = np.array(image, dtype="uint8")
    np_image = np_image / 255.0
    np_image = np_image.reshape(1, 224, 224, 3)
    return np_image


#####


def check_status(app_name):
    sage_client = boto3.client("sagemaker", region_name=region)
    endpoint_description = sage_client.describe_endpoint(EndpointName=app_name)
    endpoint_status = endpoint_description["EndpointStatus"]
    return endpoint_status


def query_endpoint(app_name, data):
    client = boto3.session.Session().client("sagemaker-runtime", region)

    response = client.invoke_endpoint(
        EndpointName=app_name,
        Body=data,
        ContentType="application/json",
    )

    preds = response["Body"].read().decode("ascii")
    preds = json.loads(preds)
    print("Received response: {}".format(preds))
    return preds


# Check endpoint status
print("[+] Endpoint Status")
print("Application status is {}".format(check_status(app_name)))

# Prepare to give for predictions
# iris = datasets.load_iris()
# x = iris.data[:, 2:]
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=19890528)
# query_input = pd.DataFrame(X_train).iloc[[15]].to_json(orient='split')
# print(query_input)


# file = open(
#     "/Users/sebastian.blum/Documents/Personal/mlops-airflow-DAGs/cnn_skin_cancer/images/inferencing/1.jpg",
#     "r",
# )
# print(file)
# print(type(file))

file = "/Users/sebastian.blum/Documents/Personal/mlops-airflow-DAGs/cnn_skin_cancer/images/inferencing/1.jpg"


image = _read_imagefile(file)

print("[+] Preprocess Data")
np_image = _preprocess_image(image)
# print(np_image)
# Create test data and make inference from endpoint

payload = json.dumps(np_image.tolist())
# payload = np_image.tolist()
# print(payload)


payload = json.dumps({"instances": np_image.tolist()})

print("[+] Prediction")
predictions = query_endpoint(app_name=app_name, data=payload)
