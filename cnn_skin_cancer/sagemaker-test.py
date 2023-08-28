import mlflow
import mlflow.sagemaker
from mlflow import MlflowClient

# Set vars

# mlflow sagemaker build-and-push-container

MLFLOW_TRACKING_URI_local = "http://127.0.0.1:5007/"
tag_id = "2.4.1"
region_name = "eu-central-1"
aws_id = "855372857567"
role_name = "test-role-sagemaker"

model_name = "Basic"
experiment_name = "cnn_skin_cancer"

instance_type = "ml.t2.large"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_local)

# Get model information

experiment_id = dict(mlflow.get_experiment_by_name(experiment_name))["experiment_id"]

client = mlflow.MlflowClient()
model_version_details = client.get_model_version(
    name=model_name,
    version="6",
)

# print(model_version_details)
run_id = model_version_details.run_id
source = model_version_details.source

print(f"experiment_id:  {experiment_id}")
print(f"run_id:         {run_id}")
print(f"source:         {source}")


# deploy to sagemaker

# URL of the ECR-hosted Docker image the model should be deployed into
image_url = "<YOUR mlflow-pyfunc ECR IMAGE URI>"
image_url = f"{aws_id}.dkr.ecr.{region_name}.amazonaws.com/mlflow-pyfunc:{tag_id}"

# The location, in URI format, of the MLflow model to deploy to SageMaker.
model_uri = "<YOUR MLFLOW MODEL LOCATION>"
model_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/{model_name}"
model_uri = "mlflow-artifacts:/768833903712672770/5014601c86a14265b887c95eaefeeda3/artifacts/model"

# mlflow.build_and_push_container


endpoint_name = "test-cnn-skin-cancer"

mlflow.sagemaker._deploy(
    mode="create",
    app_name=endpoint_name,
    model_uri=model_uri,
    image_url=image_url,
    execution_role_arn=f"arn:aws:iam::{aws_id}:role/{role_name}",
    instance_type=instance_type,
    instance_count=1,
    region_name=region_name,
    timeout_seconds=2400,
)
