# import pendulum
# from airflow.decorators import dag, task
# from airflow.operators.bash import BashOperator


# ##### AIRFLOW DAG
# #
# @dag(
#     "cnn_skin_cancer_training_pipeline",
#     default_args={
#         "owner": "seblum",
#         "depends_on_past": False,
#         "start_date": pendulum.datetime(2021, 1, 1, tz="Europe/Amsterdam"),
#         "tags": ["Keras CNN to classify skin cancer"],
#     },
#     schedule_interval=None,
#     max_active_runs=1,
# )
# def cnn_skin_cancer_workflow():
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
