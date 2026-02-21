"""
Docker-based Airflow DAG for local development of the CNN skin cancer pipeline.

Secrets are resolved from the Airflow Kubernetes Secrets Backend when running on
Kubernetes, or from environment variables for local Docker-based development.
"""

import os
from enum import Enum

import mlflow
import pendulum
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator

MLFLOW_TRACKING_URI_LOCAL = "http://127.0.0.1:5008/"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5008")
EXPERIMENT_NAME = "cnn_skin_cancer"
skin_cancer_container_image = "seblum/cnn-skin-cancer-model:latest"

# Credentials are consumed from the container environment (e.g. Docker .env file or
# Kubernetes secrets mounted via ESO). Never hardcoded here.
_docker_env = {
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "AWS_BUCKET": os.getenv("AWS_BUCKET", ""),
    "AWS_REGION": os.getenv("AWS_REGION", ""),
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", ""),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    "AWS_ROLE_NAME": os.getenv("AWS_ROLE_NAME", ""),
}

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_LOCAL)


def make_mlflow() -> str:
    try:
        mlflow.create_experiment(EXPERIMENT_NAME)
    except Exception:
        pass
    return mlflow.set_experiment(EXPERIMENT_NAME).experiment_id


mlflow_experiment_id = make_mlflow()


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


@dag(
    dag_id="cnn_skin_cancer_docker_workflow",
    default_args={
        "owner": "seblum",
        "depends_on_past": False,
        "start_date": pendulum.datetime(2021, 1, 1, tz="UTC"),
    },
    tags=["Keras CNN to classify skin cancer"],
    schedule=None,
    max_active_runs=1,
)
def cnn_skin_cancer_docker_workflow():
    """Local Docker-based pipeline for development and testing."""

    _docker_task_defaults = {
        "image": skin_cancer_container_image,
        "environment": _docker_env,
        "working_dir": "/app",
        "force_pull": True,
        "network_mode": "bridge",
        "multiple_outputs": True,
    }

    @task.docker(**_docker_task_defaults)
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

    @task.docker(**_docker_task_defaults)
    def model_training_op(mlflow_experiment_id: str, model_class: str, model_params: dict, input: dict) -> dict:
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

    @task.docker(**_docker_task_defaults)
    def compare_models_op(train_data_basic: dict, train_data_resnet50: dict, train_data_crossval: dict) -> dict:
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

    serve_fastapi = BashOperator(
        task_id="fastapi-serve-app",
        bash_command=(
            'docker run --detach -p 80:80 seblum/model-serving:fastapi-serve-app '
            '&& echo "fastapi-serve running"'
        ),
    )

    serve_streamlit = BashOperator(
        task_id="streamlit-inference-app",
        bash_command=(
            'docker run --detach -p 8501:8501 seblum/model-serving:streamlit-inference-app '
            '&& echo "streamlit-inference running"'
        ),
    )

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
    compare_result >> serve_fastapi >> serve_streamlit


cnn_skin_cancer_docker_workflow()
