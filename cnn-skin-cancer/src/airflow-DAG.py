from cnn_skin_cancer.Docker.preprocessing.app.preprocessing import run_preprocessing
from cnn_skin_cancer.src.model.basic_model import train_basic_model
from cnn_skin_cancer.src.model.crossval_model import train_crossval_model
from cnn_skin_cancer.src.model.resnet50_model import train_resnet50_model
from cnn_skin_cancer.src.compare_models import compare_models

import pendulum
import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.docker_operator import DockerOperator

# SET MLFLOW

MLFLOW_TRACKING_URI = "http://127.0.0.1:5008/"
EXPERIMENT_NAME = "cnn_skin_cancer"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

try:
    # Creating an experiment
    mlflow_experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
except:
    pass
# Setting the environment with the created experiment
mlflow_experiment_id = mlflow.set_experiment(EXPERIMENT_NAME).experiment_id

# SET AIRFLOW

# Create Airflow DAG
default_args = {
    "owner": "seblum",
    "depends_on_past": False,
    "start_date": pendulum.datetime(2021, 1, 1, tz="UTC"),
    "provide_context": True,
    "tags": ["Keras CNN to classify skin cancer"],
}
dag = DAG("cnn_skin_cancer", default_args=default_args, schedule_interval=None, max_active_runs=1)

# KubernetesPodOperator
# https://stackoverflow.com/questions/65858358/how-to-run-a-python-function-or-script-somescript-py-on-the-kubernetespodoperato
run_preprocessing_op = PythonOperator(
    task_id="run_preprocessing",
    provide_context=True,
    python_callable=run_preprocessing,
    op_kwargs={"mlflow_tracking_uri": MLFLOW_TRACKING_URI, "mlflow_experiment_id": mlflow_experiment_id},
    dag=dag,
)

op_kwargs_basic_model = {"mlflow_tracking_uri": MLFLOW_TRACKING_URI, "mlflow_experiment_id": mlflow_experiment_id}
train_basic_model_op = PythonOperator(
    task_id="train_basic_model",
    provide_context=True,
    op_kwargs=op_kwargs_basic_model,
    python_callable=train_basic_model,
    dag=dag,
)

train_crossval_model_op = PythonOperator(
    task_id="train_crossval_model",
    provide_context=True,
    op_kwargs={"mlflow_tracking_uri": MLFLOW_TRACKING_URI, "mlflow_experiment_id": mlflow_experiment_id},
    python_callable=train_crossval_model,
    dag=dag,
)

train_resnet50_op = PythonOperator(
    task_id="train_resnet50_model",
    provide_context=True,
    op_kwargs={"mlflow_tracking_uri": MLFLOW_TRACKING_URI, "mlflow_experiment_id": mlflow_experiment_id},
    python_callable=train_resnet50_model,
    dag=dag,
)

compare_models_op = PythonOperator(
    task_id="compare_models",
    provide_context=True,
    op_kwargs={"mlflow_tracking_uri": MLFLOW_TRACKING_URI},
    python_callable=compare_models,
    dag=dag,
)

serve_fastapi_app_op = DockerOperator(
    task_id="serve_fastapi_app",
    provide_context=True,
    # xcom.pull serving_model_name
    environment={"MLFLOW_TRACKING_URI": "host.docker.internal:5008", "MLFLOW_MODEL_NAME": "basic-keras-cnn", "MLFLOW_MODEL_VERSION": 5},
    image='seblum/model-serving:fastapi-serve',
    container_name='fastapi-serve',
    api_version='auto',
    auto_remove=True,
    #command="echo hello",
    #docker_url="unix://var/run/docker.sock", # default
    #network_mode="bridge",
    dag=dag,
)

serve_streamlit_app_op = DockerOperator(
    task_id="serve_streamlit_app",
    provide_context=True,
    environment={"FASTAPI_SERVING_IP": "host.docker.internal", "FASTAPI_SERVING_PORT": 80},
    image='seblum/model-serving:streamlit-inference',
    container_name='streamlit-inference',
    api_version='auto',
    auto_remove=True,
    #command="echo hello",
    #docker_url="unix://var/run/docker.sock", # default
    #network_mode="bridge",
    dag=dag,
)

# set task dependencies
run_preprocessing_op >> [train_basic_model_op,train_crossval_model_op,train_resnet50_op]

[train_basic_model_op,train_crossval_model_op,train_resnet50_op] >> compare_models_op

compare_models_op >> [serve_streamlit_app_op,serve_fastapi_app_op]
