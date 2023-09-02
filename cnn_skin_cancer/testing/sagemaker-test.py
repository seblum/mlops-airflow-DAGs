# import mlflow
# import mlflow.sagemaker
# from mlflow import MlflowClient

# # Set vars

# # mlflow sagemaker build-and-push-container

# MLFLOW_TRACKING_URI_local = "http://127.0.0.1:5007/"
# tag_id = "2.4.1"
# tag_id = "v2.3.2"
# region_name = "eu-central-1"
# aws_id = "ID"

# role_name = "sagemaker-access-role"
# role_name = "sagemakertestrole"

# model_name = "Basic"
# experiment_name = "cnn_skin_cancer"

# instance_type = "ml.t2.large"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_local)

# repository_name = "mlflow-pyfunc"
# repository_name = "mlflow-sagemaker-deployment"
# model_version = 6

# # Get model information


# from pathlib import Path


# def make_path():
#     path = f"{Path(__file__).parent.parent}/images"
#     return path


# def get_mlflow_parameters(
#     experiment_name: str, model_name: str, model_version: str
# ) -> (str, str, str):
#     experiment_id = dict(mlflow.get_experiment_by_name(experiment_name))[
#         "experiment_id"
#     ]

#     client = mlflow.MlflowClient()
#     model_version_details = client.get_model_version(
#         name=model_name,
#         version=model_version,
#     )

#     # print(model_version_details)
#     run_id = model_version_details.run_id
#     source = model_version_details.source

#     return experiment_id, run_id, source


# experiment_id, run_id, source = get_mlflow_parameters(
#     experiment_name=experiment_name, model_name=model_name, model_version=model_version
# )

# print(f"experiment_id:  {experiment_id}")
# print(f"run_id:         {run_id}")
# print(f"source:         {source}")

# print(f"{source.removesuffix(model_name)}model")

# # deploy to sagemaker

# # URL of the ECR-hosted Docker image the model should be deployed into
# image_url = "<YOUR mlflow-pyfunc ECR IMAGE URI>"
# image_url = f"{aws_id}.dkr.ecr.{region_name}.amazonaws.com/{repository_name}:{tag_id}"

# # The location, in URI format, of the MLflow model to deploy to SageMaker.
# model_uri = "<YOUR MLFLOW MODEL LOCATION>"
# model_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/{model_name}"
# model_uri = "mlflow-artifacts:/768833903712672770/5014601c86a14265b887c95eaefeeda3/artifacts/model"

# # mlflow.build_and_push_container


# endpoint_name = "test-cnn-skin-cancer"

# import boto3

# session = boto3.Session()
# sts = session.client("sts")
# response = sts.assume_role(
#     RoleArn="arn:aws:iam::855372857567:role/sagemakertestrole",
#     RoleSessionName="learnaws-test-session",
# )
# print(response)

# mlflow.sagemaker._deploy(
#     mode="create",
#     app_name=endpoint_name,
#     model_uri=model_uri,
#     image_url=image_url,
#     execution_role_arn=f"arn:aws:iam::{aws_id}:role/{role_name}",
#     instance_type=instance_type,
#     instance_count=1,
#     region_name=region_name,
#     timeout_seconds=2400,
# )
