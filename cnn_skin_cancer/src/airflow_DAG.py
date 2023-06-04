import json

# actually not needed anymore since there is the docker operator now
import os
from datetime import datetime

import mlflow
import numpy as np
import pendulum
from airflow.decorators import dag, task
from keras.utils.np_utils import (  # used for converting labels to one-hot-encoding
    to_categorical,
)
from sklearn.utils import shuffle
from tqdm import tqdm

# from .model.utils import Model_Class
from cnn_skin_cancer.src.model.utils import Model_Class

# SET MLFLOW

MLFLOW_TRACKING_URI_local = "http://127.0.0.1:5008/"
MLFLOW_TRACKING_URI = "http://host.docker.internal:5008"
EXPERIMENT_NAME = "cnn_skin_cancer"
AWS_BUCKET = os.getenv("AWS_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ROLE_NAME = os.getenv("AWS_ROLE_NAME")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_local)

try:
    # Creating an experiment
    mlflow_experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
except:
    pass
# Setting the environment with the created experiment
mlflow_experiment_id = mlflow.set_experiment(EXPERIMENT_NAME).experiment_id

from enum import Enum


class Model_Class(Enum):
    """This enum includes different models."""

    Basic = "Basic"
    CrossVal = "CrossVal"
    ResNet50 = "ResNet50"


# SET AIRFLOW

# Set various params and args
dag_default_args = {
    "owner": "seblum",
    "depends_on_past": False,
    "start_date": pendulum.datetime(2021, 1, 1, tz="UTC"),
    # "provide_context": True,
    "tags": ["Keras CNN to classify skin cancer"],
}

## PREPROCESSING

kwargs_data_preprocessing = {
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "MLFLOW_EXPERIMENT_ID": mlflow_experiment_id,
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    "AWS_BUCKET": AWS_BUCKET,
    "AWS_ROLE_NAME": AWS_ROLE_NAME,
    "AWS_REGION": AWS_REGION,
}

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
    "pooling": "avg",  # needed for resnet50
    "verbose": 2,
}


@dag(
    "cnn_skin_cancer_taskflow",
    default_args=dag_default_args,
    schedule_interval=None,
    max_active_runs=1,
)
def tutorial_taskflow_api():
    @task.docker(
        image="seblum/cnn-model:python-base",
        multiple_outputs=True,
        environment=kwargs_data_preprocessing,
        working_dir="/app",
        force_pull=True,
        # tmp_dir="/app"
    )
    def preprocessing_op(mlflow_experiment_id, AWS_BUCKET):
        import os
        import sys

        sys.path.append(os.path.abspath("/app/"))

        from preprocessing import data_preprocessing

        (
            X_train_data_path,
            y_train_data_path,
            X_test_data_path,
            y_test_data_path,
        ) = data_preprocessing(mlflow_experiment_id=mlflow_experiment_id, aws_bucket=AWS_BUCKET)

        # Create dictionary with S3 paths to return
        return_dict = {
            "X_train_data_path": X_train_data_path,
            "y_train_data_path": y_train_data_path,
            "X_test_data_path": X_test_data_path,
            "y_test_data_path": y_test_data_path,
        }
        return return_dict

    @task.docker(
        image="seblum/cnn-model:python-base",
        multiple_outputs=True,
        environment=kwargs_data_preprocessing,
        working_dir="/app",
        force_pull=True,
    )
    def model_training_op(mlflow_experiment_id, model_class, model_params, AWS_BUCKET, input):
        import os
        import sys

        sys.path.append(os.path.abspath("/app/"))

        from train import train_model

        run_id, model_name, model_version, model_stage = train_model(
            mlflow_experiment_id=mlflow_experiment_id,
            model_class=model_class,
            model_params=model_params,
            aws_bucket=AWS_BUCKET,
            import_dict=input,
        )

        # Create dictionary with S3 paths to return
        return_dict = {
            "run_id": run_id,
            "model_name": model_name,
            "model_version": model_version,
            "model_stage": model_stage,
        }
        return return_dict

    @task.docker(
        image="seblum/cnn-model:python-base",
        multiple_outputs=True,
        environment=kwargs_data_preprocessing,
        force_pull=True,
    )
    def compare_models_op(train_data_basic, train_data_resnet50):
        import os
        import sys

        sys.path.append(os.path.abspath("/app/"))

        compare_dict = {
            train_data_basic["model_name"]: train_data_basic["run_id"],
            train_data_resnet50["model_name"]: train_data_resnet50["run_id"],
        }

        print(compare_dict)
        from compare_models import compare_models

        serving_model_name, serving_model_uri, serving_model_version = compare_models(input_dict=compare_dict)
        return_dict = {
            "serving_model_name": serving_model_name,
            "serving_model_uri": serving_model_uri,
            "serving_model_version": serving_model_version,
        }
        return return_dict

    @task.docker(
        image="seblum/model-serving:fastapi-serve",
        multiple_outputs=True,
        environment=kwargs_data_preprocessing,
        force_pull=True,
    )
    def serve_fastapi_app_op(**kwargs):
        return True

    @task.docker(
        image="seblum/model-serving:streamlit-inference",
        multiple_outputs=True,
        environment=kwargs_data_preprocessing,
    )
    def serve_streamlit_app_op(**kwargs):
        return True

    # CREATE PIPELINE

    data = preprocessing_op(
        mlflow_experiment_id=mlflow_experiment_id,
        AWS_BUCKET=AWS_BUCKET,
    )
    train_data_basic = model_training_op(
        mlflow_experiment_id=mlflow_experiment_id,
        model_class=Model_Class.Basic.name,
        model_params=model_params,
        AWS_BUCKET=AWS_BUCKET,
        input=data,
    )
    train_data_resnet50 = model_training_op(
        mlflow_experiment_id=mlflow_experiment_id,
        model_class=Model_Class.ResNet50.name,
        model_params=model_params,
        AWS_BUCKET=AWS_BUCKET,
        input=data,
    )
    compare_models_dict = compare_models_op(train_data_basic, train_data_resnet50)
    serve_fastapi_app_op(compare_models_dict)
    serve_streamlit_app_op(compare_models_dict)


tutorial_taskflow_api()
