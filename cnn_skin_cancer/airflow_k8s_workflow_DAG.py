"""
Kubernetes-based Airflow DAG for the CNN skin cancer classification pipeline.

Secret management: secrets are resolved via the Airflow Kubernetes Secrets Backend
(AIRFLOW__SECRETS__BACKEND = KubernetesSecretBackend), so explicit `Secret` objects
are no longer needed — pods pick up environment variables from the pod's service account
via IRSA and from K8s secrets synced by External Secrets Operator.
"""

from enum import Enum

import mlflow
import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from kubernetes.client import models as k8s

EXPERIMENT_NAME = "cnn_skin_cancer"
skin_cancer_container_image = "seblum/cnn-skin-cancer-model:latest"

MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
ECR_REPOSITORY_NAME = Variable.get("ECR_REPOSITORY_NAME")
ECR_SAGEMAKER_IMAGE_TAG = Variable.get("ECR_SAGEMAKER_IMAGE_TAG")

# Node selector and tolerations to schedule model training on dedicated nodes
tolerations = [k8s.V1Toleration(key="dedicated", operator="Equal", value="t3_large", effect="NoSchedule")]
node_selector = {"role": "t3_large"}


class Model_Class(Enum):
    """Supported model architectures."""

    Basic = "Basic"
    CrossVal = "CrossVal"
    ResNet50 = "ResNet50"


model_params = {
    "num_classes": 2,
    "input_shape": (224, 224, 3),
    "activation": "relu",
    "kernel_initializer_glob": "glorot_uniform",
    "kernel_initializer_norm": "normal",
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "metrics": ["accuracy"],
    "validation_split": 0.2,
    "epochs": 2,
    "batch_size": 64,
    "learning_rate": 1e-5,
    "pooling": "avg",
    "verbose": 2,
}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def make_mlflow() -> str:
    """Create or retrieve the MLflow experiment and return its ID."""
    try:
        mlflow.create_experiment(EXPERIMENT_NAME)
    except Exception:
        pass
    return mlflow.set_experiment(EXPERIMENT_NAME).experiment_id


mlflow_experiment_id = make_mlflow()

# Shared Kubernetes pod defaults for all tasks
_k8s_defaults = {
    "image": skin_cancer_container_image,
    "namespace": "airflow",
    "in_cluster": True,
    "get_logs": True,
    "do_xcom_push": True,
    "startup_timeout_seconds": 300,
    "service_account_name": "airflow-sa",
    # No explicit Secret() objects — credentials come from IRSA + K8s secrets
    # synced by External Secrets Operator and resolved by the Kubernetes Secrets Backend.
}


@dag(
    dag_id="cnn_skin_cancer_workflow",
    default_args={
        "owner": "seblum",
        "depends_on_past": False,
        "start_date": pendulum.datetime(2021, 1, 1, tz="Europe/Amsterdam"),
    },
    tags=["Keras CNN to classify skin cancer"],
    schedule=None,
    max_active_runs=1,
)
def cnn_skin_cancer_workflow():
    """Airflow DAG: train, compare, and deploy skin cancer classification models on Kubernetes."""

    @task.kubernetes(
        task_id="preprocessing_op",
        env_vars={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
        **_k8s_defaults,
    )
    def preprocessing_op(mlflow_experiment_id: str) -> dict:
        """Preprocess raw images and upload NumPy arrays to S3."""
        import os

        from src.preprocessing import data_preprocessing

        X_train, y_train, X_test, y_test = data_preprocessing(
            mlflow_experiment_id=mlflow_experiment_id,
            aws_bucket=os.environ["AWS_BUCKET"],
        )
        return {
            "X_train_data_path": X_train,
            "y_train_data_path": y_train,
            "X_test_data_path": X_test,
            "y_test_data_path": y_test,
        }

    @task.kubernetes(
        task_id="model_training_op",
        env_vars={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
        node_selector=node_selector,
        tolerations=tolerations,
        **_k8s_defaults,
    )
    def model_training_op(
        mlflow_experiment_id: str,
        model_class: str,
        model_params: dict,
        input: dict,
    ) -> dict:
        """Train a model architecture and log it to MLflow."""
        import os

        from src.train import train_model

        run_id, model_name, model_version, model_stage = train_model(
            mlflow_experiment_id=mlflow_experiment_id,
            model_class=model_class,
            model_params=model_params,
            aws_bucket=os.environ["AWS_BUCKET"],
            import_dict=input,
        )
        return {
            "run_id": run_id,
            "model_name": model_name,
            "model_version": model_version,
            "model_stage": model_stage,
        }

    @task.kubernetes(
        task_id="compare_models_op",
        env_vars={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
        **_k8s_defaults,
    )
    def compare_models_op(
        train_data_basic: dict,
        train_data_resnet50: dict,
        train_data_crossval: dict,
    ) -> dict:
        """Select the best model by accuracy and promote it to Staging."""
        from src.compare_models import compare_models

        compare_dict = {
            train_data_basic["model_name"]: train_data_basic["run_id"],
            train_data_resnet50["model_name"]: train_data_resnet50["run_id"],
            train_data_crossval["model_name"]: train_data_crossval["run_id"],
        }
        serving_name, serving_uri, serving_version = compare_models(input_dict=compare_dict)
        return {
            "serving_model_name": serving_name,
            "serving_model_uri": serving_uri,
            "serving_model_version": serving_version,
        }

    @task.kubernetes(
        task_id="deploy_model_to_sagemaker_op",
        env_vars={
            "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
            "ECR_REPOSITORY_NAME": ECR_REPOSITORY_NAME,
            "ECR_SAGEMAKER_IMAGE_TAG": ECR_SAGEMAKER_IMAGE_TAG,
        },
        **{k: v for k, v in _k8s_defaults.items() if k != "do_xcom_push"},
    )
    def deploy_model_to_sagemaker_op(serving_model_dict: dict) -> None:
        """Deploy the winning model to an Amazon SageMaker endpoint."""
        from src.deploy_model_to_sagemaker import deploy_model_to_sagemaker

        result = deploy_model_to_sagemaker(
            mlflow_model_name=serving_model_dict["serving_model_name"],
            mlflow_model_uri=serving_model_dict["serving_model_uri"],
            mlflow_model_version=serving_model_dict["serving_model_version"],
            mlflow_experiment_name=EXPERIMENT_NAME,
            sagemaker_endpoint_name="test-cnn-skin-cancer",
            sagemaker_instance_type="ml.t2.large",
        )
        print(f"Deploy result: {result}")

    # ── Pipeline wiring ───────────────────────────────────────────────────────
    preprocessed_data = preprocessing_op(mlflow_experiment_id=mlflow_experiment_id)

    train_data_basic = model_training_op(
        mlflow_experiment_id=mlflow_experiment_id,
        model_class=Model_Class.Basic.name,
        model_params=model_params,
        input=preprocessed_data,
    )
    train_data_resnet50 = model_training_op(
        mlflow_experiment_id=mlflow_experiment_id,
        model_class=Model_Class.ResNet50.name,
        model_params=model_params,
        input=preprocessed_data,
    )
    train_data_crossval = model_training_op(
        mlflow_experiment_id=mlflow_experiment_id,
        model_class=Model_Class.CrossVal.name,
        model_params=model_params,
        input=preprocessed_data,
    )

    compare_result = compare_models_op(train_data_basic, train_data_resnet50, train_data_crossval)
    deploy_model_to_sagemaker_op(compare_result)


cnn_skin_cancer_workflow()
