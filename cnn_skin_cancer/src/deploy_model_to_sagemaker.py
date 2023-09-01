import os

import mlflow


def deploy_model_to_sagemaker(
    mlflow_model_name: str,
    mlflow_experiment_name: str,
    mlflow_model_version: int,
    sagemaker_endpoint_name: str,
    sagemaker_instance_type: str,
) -> bool:
    """
    Deploy a machine learning model to AWS SageMaker from MLflow.

    This function deploys an MLflow model to an AWS SageMaker endpoint using the specified configuration.

    Args:
        mlflow_model_name (str): The name of the MLflow model to deploy.
        mlflow_experiment_name (str): The name of the MLflow experiment containing the model.
        mlflow_model_version (int): The version of the MLflow model to deploy.
        sagemaker_endpoint_name (str): The desired name for the SageMaker endpoint.
        sagemaker_instance_type (str): The SageMaker instance type for deployment.

    Returns:
        bool: True if the task was completed successfully, otherwise False.
    """
    # Retrieve AWS and MLflow environment variables
    AWS_ID = os.getenv("AWS_ID")
    ECR_REPOSITORY_NAME = os.getenv("ECR_REPOSITORY_NAME")
    SAGEMAKER_ACCESS_ROLE_ARN = os.getenv("SAGEMAKER_ACCESS_ROLE_ARN")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    AWS_REGION = os.getenv("AWS_REGION")
    ECR_SAGEMAKER_IMAGE_TAG = os.getenv("ECR_SAGEMAKER_IMAGE_TAG")

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def _build_image_url(
        aws_id: str,
        aws_region: str,
        ecr_repository_name: str,
        ecr_sagemaker_image_tag: str,
    ) -> str:
        """
        Build the ECR image URL for SageMaker.

        Args:
            aws_id (str): AWS account ID.
            aws_region (str): AWS region.
            ecr_repository_name (str): Name of the ECR repository.
            ecr_sagemaker_image_tag (str): Tag for the ECR image.

        Returns:
            str: The ECR image URL for SageMaker.
        """
        image_url = f"{aws_id}.dkr.ecr.{aws_region}.amazonaws.com/{ecr_repository_name}:{ecr_sagemaker_image_tag}"
        return image_url

    def _build_execution_role_arn(aws_id: str, sagemaker_access_role_arn: str) -> str:
        """
        Build the SageMaker execution role ARN.

        Args:
            aws_id (str): AWS account ID.
            sagemaker_access_role_arn (str): SageMaker execution role name.

        Returns:
            str: The SageMaker execution role ARN.
        """
        execution_role_arn = f"arn:aws:iam::{aws_id}:role/{sagemaker_access_role_arn}"
        return execution_role_arn

    def _get_mlflow_parameters(experiment_name: str, model_name: str, model_version: int) -> (str, str, str):
        """
        Retrieve MLflow model parameters.

        Args:
            experiment_name (str): Name of the MLflow experiment.
            model_name (str): Name of the MLflow model.
            model_version (int): Version of the MLflow model.

        Returns:
            Tuple[str, str]: The model URI and source.
        """
        experiment_id = dict(mlflow.get_experiment_by_name(experiment_name))["experiment_id"]

        client = mlflow.MlflowClient()
        model_version_details = client.get_model_version(
            name=model_name,
            version=model_version,
        )

        run_id = model_version_details.run_id
        model_uri = f"mlruns/{experiment_id}/{run_id}/artifacts/{model_name}"
        model_source = model_version_details.source

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
    return True
