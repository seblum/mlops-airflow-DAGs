import os

import mlflow


def deploy_model_to_sagemaker(
    mlflow_model_name: str,
    mlflow_experiment_name: str,
    mlflow_model_version: int,
    sagemaker_endpoint_name: int,
    sagemaker_instance_type: str,
):
    AWS_ID = os.getenv("AWS_ID")
    ECR_REPOSITORY_NAME = os.getenv("ECR_REPOSITORY_NAME")
    SAGEMAKER_ACCESS_ROLE_ARN = os.getenv("SAGEMAKER_ACCESS_ROLE_ARN")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    AWS_REGION = os.getenv("AWS_REGION")
    ECR_SAGEMAKER_IMAGE_TAG = os.getenv("ECR_SAGEMAKER_IMAGE_TAG")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def _build_image_url(
        aws_id: str,
        aws_region: str,
        ecr_repository_name: str,
        ecr_sagemaker_image_tag: str,
    ) -> str:
        image_url = f"{aws_id}.dkr.ecr.{aws_region}.amazonaws.com/{ecr_repository_name}:{ecr_sagemaker_image_tag}"
        return image_url

    def _build_execution_role_arn(aws_id: str, sagemaker_access_role_arn: str) -> str:
        execution_role_arn = f"arn:aws:iam::{aws_id}:role/{sagemaker_access_role_arn}"
        return execution_role_arn

    def _get_mlflow_parameters(experiment_name: str, model_name: str, model_version: int) -> (str, str, str):
        experiment_id = dict(mlflow.get_experiment_by_name(experiment_name))["experiment_id"]

        client = mlflow.MlflowClient()
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version,
        )

        run_id = model_version_details.run_id
        model_source = model_version_details.source
        model_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/{model_name}"

        return model_uri, model_source

    image_url = _build_image_url(
        aws_id=AWS_ID,
        aws_region=AWS_REGION,
        ecr_repository_name=ECR_REPOSITORY_NAME,
        ecr_sagemaker_image_tag=ECR_SAGEMAKER_IMAGE_TAG,
    )
    execution_role_arn = _build_execution_role_arn(aws_id=AWS_ID, sagemaker_access_role_arn=SAGEMAKER_ACCESS_ROLE_ARN)
    model_uri, model_source = _get_mlflow_parameters(
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
