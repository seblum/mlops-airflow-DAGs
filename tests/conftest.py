"""
Shared fixtures for DAG tests.

Patches external dependencies (MLflow, Airflow Variables) so DAG files can be
imported without a running Airflow environment or MLflow server.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def patch_external_calls():
    """Prevent real network calls when DAG modules are imported."""
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "test-experiment-id"

    with (
        patch("mlflow.set_tracking_uri"),
        patch("mlflow.create_experiment", return_value="test-experiment-id"),
        patch("mlflow.set_experiment", return_value=mock_experiment),
        patch("airflow.models.Variable.get", side_effect=_variable_get),
    ):
        yield


def _variable_get(key: str, default_var: str = "") -> str:
    """Return predictable test values for Airflow Variables."""
    defaults = {
        "MLFLOW_TRACKING_URI": "http://mlflow-test:5000",
        "ECR_REPOSITORY_NAME": "test-ecr-repo",
        "ECR_SAGEMAKER_IMAGE_TAG": "v1.0.0",
    }
    return defaults.get(key, default_var)
