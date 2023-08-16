import pendulum
import yaml
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor


##### AIRFLOW DAG
#
@dag(
    "cnn_skin_cancer_training_pipeline",
    default_args={
        "owner": "seblum",
        "depends_on_past": False,
        "start_date": pendulum.datetime(2021, 1, 1, tz="Europe/Amsterdam"),
        "tags": ["Keras CNN to classify skin cancer"],
    },
    schedule_interval=None,
    max_active_runs=1,
)
def cnn_skin_cancer_deployment():
    trigger_deploy = ExternalTaskSensor(
        task_id="trigger_deploy",
        external_dag_id="cnn_skin_cancer_training_pipeline",
        external_task_id="compare-models",
        start_date=pendulum.datetime(2021, 1, 1, tz="Europe/Amsterdam"),
        # execution_delta=timedelta(hours=1),
        # timeout=3600,
    )

    @task(
        name="deploy_model",
        # namespace="seldon-core",
        # env_vars={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
        # in_cluster=True,
        # get_logs=True,
        # do_xcom_push=True,
        # startup_timeout_seconds=300,
        # service_account_name="airflow-sa",
    )
    def deploy_model():
        # set yaml
        # kubectl yaml
        pass

    trigger_deploy
    deploy_model
    #     @task.kubernetes(
    #         image=skin_cancer_container_image,
    #         name="preprocessing",
    #         namespace="airflow",
    #         env_vars={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
    #         in_cluster=True,
    #         get_logs=True,
    #         do_xcom_push=True,
    #         startup_timeout_seconds=300,
    #         # service_account_name="airflow-sa",
    #         secrets=[
    #             SECRET_AWS_BUCKET,
    #             SECRET_AWS_REGION,
    #             SECRET_AWS_ACCESS_KEY_ID,
    #             SECRET_AWS_SECRET_ACCESS_KEY,
    #             SECRET_AWS_ROLE_NAME,
    #         ],
    #     )
    #     def deployment_op(mlflow_experiment_id: str) -> dict:
    #         """
    #         Perform data preprocessing.

    #         Args:
    #             mlflow_experiment_id (str): The MLflow experiment ID.

    #         Returns:
    #             dict: A dictionary containing the paths to preprocessed data.
    #         """
    #         import os

    #         # import time
    #         # time.sleep(60)

    #         aws_bucket = os.getenv("AWS_BUCKET")

    #         from src.preprocessing import data_preprocessing

    #         (
    #             X_train_data_path,
    #             y_train_data_path,
    #             X_test_data_path,
    #             y_test_data_path,
    #         ) = data_preprocessing(mlflow_experiment_id=mlflow_experiment_id, aws_bucket=aws_bucket)

    #         # Create dictionary with S3 paths to return
    #         return_dict = {
    #             "X_train_data_path": X_train_data_path,
    #             "y_train_data_path": y_train_data_path,
    #             "X_test_data_path": X_test_data_path,
    #             "y_test_data_path": y_test_data_path,
    #         }
    #         return return_dict

    #     # serve_streamlit_app_op = BashOperator(
    #     #     task_id="streamlit-inference-app",
    #     #     bash_command='docker run --detach -p 8501:8501 -it seblum/model-serving:streamlit-inference-app && echo "streamlit-inference running"',
    #     # )

    #     # CREATE PIPELINE

    #     deployment_op()
    #     # compare_models_dict >> serve_fastapi_app_op >> serve_streamlit_app_op

    # cnn_skin_cancer_workflow()

    seldon_deployment = {
        "apiVersion": "machinelearning.seldon.io/v1alpha2",
        "kind": "SeldonDeployment",
        "metadata": {"name": "mlflow", "namespace": "seldon-system"},
        "spec": {
            "protocol": "v2",
            "name": "wines",
            "predictors": [
                {
                    "graph": {
                        "children": [],
                        "implementation": "MLFLOW_SERVER",
                        "modelUri": "s3://d7k27cmkytac-mlflow-artifact-bucket/test/elasticnet_wine_44eb4bd043964a34be556172a710bc18",
                        "name": "classifier",
                    },
                    "name": "default",
                    "replicas": 1,
                }
            ],
        },
    }

    print("The python dictionary is:")
    print(seldon_deployment)
    yaml_string = yaml.dump(seldon_deployment)
    print("The YAML string is:")
    print(yaml_string)


cnn_skin_cancer_deployment()
