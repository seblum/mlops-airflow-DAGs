import os
from enum import Enum

import pendulum
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator

MLFLOW_TRACKING_URI_local = "http://127.0.0.1:5008/"
MLFLOW_TRACKING_URI = "http://host.docker.internal:5008"
EXPERIMENT_NAME = "cnn_skin_cancer"
AWS_BUCKET = os.getenv("AWS_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_ROLE_NAME = os.getenv("AWS_ROLE_NAME")


# import mlflow
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_local)

# try:
#     # Creating an experiment
#     mlflow_experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
# except:
#     pass
# # Setting the environment with the created experiment
# mlflow_experiment_id = mlflow.set_experiment(EXPERIMENT_NAME).experiment_id


class Model_Class(Enum):
    """This enum includes different models."""

    Basic = "Basic"
    CrossVal = "CrossVal"
    ResNet50 = "ResNet50"


# Set various model params and airflow or environment args

dag_default_args = {
    "owner": "seblum",
    "depends_on_past": False,
    "start_date": pendulum.datetime(2021, 1, 1, tz="UTC"),
    "tags": ["Keras CNN to classify skin cancer"],
}

kwargs_env_data = {
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,
    "MLFLOW_EXPERIMENT_ID": mlflow_experiment_id,
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
    "AWS_BUCKET": AWS_BUCKET,
    "AWS_REGION": AWS_REGION,
    "AWS_ROLE_NAME": AWS_ROLE_NAME,
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

skin_cancer_container_image = "seblum/cnn-skin-cancer:latest"


@dag(
    "cnn_skin_cancer_docker_workflow",
    default_args=dag_default_args,
    schedule_interval=None,
    max_active_runs=1,
)
def cnn_skin_cancer_workflow():
    @task.docker(
        image=skin_cancer_container_image,
        multiple_outputs=True,
        environment=kwargs_env_data,
        working_dir="/app",
        force_pull=True,
        network_mode="bridge",
    )
    def preprocessing_op(mlflow_experiment_id):
        """
        Perform data preprocessing.

        Args:
            mlflow_experiment_id (str): The MLflow experiment ID.

        Returns:
            dict: A dictionary containing the paths to preprocessed data.
        """
        import os

        from src.preprocessing import data_preprocessing

        aws_bucket = os.getenv("AWS_BUCKET")

        (
            X_train_data_path,
            y_train_data_path,
            X_test_data_path,
            y_test_data_path,
        ) = data_preprocessing(mlflow_experiment_id=mlflow_experiment_id, aws_bucket=aws_bucket)

        # Create dictionary with S3 paths to return
        return_dict = {
            "X_train_data_path": X_train_data_path,
            "y_train_data_path": y_train_data_path,
            "X_test_data_path": X_test_data_path,
            "y_test_data_path": y_test_data_path,
        }
        return return_dict

    @task.docker(
        image=skin_cancer_container_image,
        multiple_outputs=True,
        environment=kwargs_env_data,
        working_dir="/app",
        force_pull=True,
        network_mode="bridge",
    )
    def model_training_op(mlflow_experiment_id, model_class, model_params, input):
        """
        Train a model.

        Args:
            mlflow_experiment_id (str): The MLflow experiment ID.
            model_class (str): The class of the model to train.
            model_params (dict): A dictionary containing the model parameters.
            input (dict): A dictionary containing the input data.

        Returns:
            dict: A dictionary containing the results of the model training.
        """
        import os

        from src.train import train_model

        aws_bucket = os.getenv("AWS_BUCKET")
        run_id, model_name, model_version, model_stage = train_model(
            mlflow_experiment_id=mlflow_experiment_id,
            model_class=model_class,
            model_params=model_params,
            aws_bucket=aws_bucket,
            import_dict=input,
        )

        return_dict = {
            "run_id": run_id,
            "model_name": model_name,
            "model_version": model_version,
            "model_stage": model_stage,
        }
        return return_dict

    @task.docker(
        image=skin_cancer_container_image,
        multiple_outputs=True,
        environment=kwargs_env_data,
        force_pull=True,
        network_mode="bridge",
    )
    def compare_models_op(train_data_basic, train_data_resnet50, train_data_crossval):
        """
        Compare trained models.

        Args:
            train_data_basic (dict): A dictionary containing the results of training the basic model.
            train_data_resnet50 (dict): A dictionary containing the results of training the ResNet50 model.
            train_data_crossval (dict): A dictionary containing the results of training the CrossVal model.

        Returns:
            dict: A dictionary containing the results of the model comparison.
        """
        compare_dict = {
            train_data_basic["model_name"]: train_data_basic["run_id"],
            train_data_resnet50["model_name"]: train_data_resnet50["run_id"],
            train_data_crossval["model_name"]: train_data_crossval["run_id"],
        }

        print(compare_dict)
        from src.compare_models import compare_models

        serving_model_name, serving_model_uri, serving_model_version = compare_models(input_dict=compare_dict)
        return_dict = {
            "serving_model_name": serving_model_name,
            "serving_model_uri": serving_model_uri,
            "serving_model_version": serving_model_version,
        }
        return return_dict

    # serve_fastapi_app_op = BashOperator(
    #     task_id="fastapi-serve-app",
    #     bash_command='docker run --detach -p 80:80 -it seblum/model-serving:fastapi-serve-app && echo "fastapi-serve running"',
    # )

    # serve_streamlit_app_op = BashOperator(
    #     task_id="streamlit-inference-app",
    #     bash_command='docker run --detach -p 8501:8501 -it seblum/model-serving:streamlit-inference-app && echo "streamlit-inference running"',
    # )

    # CREATE PIPELINE

    preprocessed_data = preprocessing_op(
        mlflow_experiment_id=mlflow_experiment_id,
    )
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
    compare_models_dict = compare_models_op(train_data_basic, train_data_resnet50, train_data_crossval)

    # compare_models_dict >> serve_fastapi_app_op >> serve_streamlit_app_op


cnn_skin_cancer_workflow()
