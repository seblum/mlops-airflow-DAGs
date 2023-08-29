import mlflow
import mlflow.sagemaker
from airflow.models import Variable
from mlflow import MlflowClient
from src.deploy_model_to_sagemaker import (
    build_execution_role_arn,
    build_image_url,
    get_mlflow_parameters,
)

AWS_ID = Variable.get("AWS_ID")
ECR_REPOSITORY_NAME = Variable.get("ECR_REPOSITORY_NAME")
SAGEMAKER_ACCESS_ROLE_ARN = Variable.get("SAGEMAKER_ACCESS_ROLE_ARN")
MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
AWS_REGION = Variable.get("AWS_REGION")
ECR_SAGEMAKER_IMAGE_TAG = Variable.get("ECR_SAGEMAKER_IMAGE_TAG")

# Set vars
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow_model_name = "Basic"
mlflow_experiment_name = "cnn_skin_cancer"
mlflow_model_version = 6

sagemaker_instance_type = "ml.t2.large"
sagemaker_endpoint_name = "test-cnn-skin-cancer"


image_url = build_image_url(
    aws_id=AWS_ID,
    aws_region=AWS_REGION,
    ecr_repository_name=ECR_REPOSITORY_NAME,
    ecr_sagemaker_image_tag=ECR_SAGEMAKER_IMAGE_TAG,
)
execution_role_arn = build_execution_role_arn(aws_id=AWS_ID, sagemaker_access_role_arn=SAGEMAKER_ACCESS_ROLE_ARN)
model_uri, model_source = get_mlflow_parameters(
    experiment_name=mlflow_experiment_name,
    model_name=mlflow_model_name,
    model_version=mlflow_model_version,
)

print(f"model_uri: {model_uri}")
print(f"model_source: {model_source}")

mlflow.sagemaker._deploy(
    mode="create",
    app_name=sagemaker_endpoint_name,
    model_uri=model_uri,
    image_url=image_url,
    execution_role_arn=execution_role_arn,
    instance_type=sagemaker_instance_type,
    instance_count=1,
    region_name=AWS_REGION,
    timeout_seconds=2400,
)
