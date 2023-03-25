from cnn_skin_cancer.model.preprocessing import run_preprocessing
from cnn_skin_cancer.model.basic_model import train_basic_model
from cnn_skin_cancer.model.crossval_model import train_crossval_model
from cnn_skin_cancer.model.resnet50_model import train_resnet50_model
from cnn_skin_cancer.model.serve_model import serve_model
from cnn_skin_cancer.model.compare_models import compare_models
import pendulum
import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator

# SET MLFLOW

mlflow_tracking_uri = "http://127.0.0.1:5008/"
experiment_name = "cnn_skin_cancer"

mlflow.set_tracking_uri(mlflow_tracking_uri)

try:
    # Creating an experiment
    mlflow_experiment_id = mlflow.create_experiment(experiment_name)
except:
    pass
# Setting the environment with the created experiment
mlflow_experiment_id = mlflow.set_experiment(experiment_name).experiment_id

# mlflow_experiment_id.experiment_id

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

# # KubernetesPodOperator
# run_preprocessing_op = PythonOperator(
#     task_id="run_preprocessing",
#     provide_context=True,
#     python_callable=run_preprocessing,
#     op_kwargs={"mlflow_tracking_uri": mlflow_tracking_uri, "mlflow_experiment_id": mlflow_experiment_id},
#     dag=dag,
# )

# train_basic_model_op = PythonOperator(
#     task_id="train_basic_model",
#     provide_context=True,
#     op_kwargs={"mlflow_tracking_uri": mlflow_tracking_uri, "mlflow_experiment_id": mlflow_experiment_id},
#     python_callable=train_basic_model,
#     dag=dag,
# )

# train_crossval_model_op = PythonOperator(
#     task_id="train_crossval_model",
#     provide_context=True,
#     op_kwargs={"mlflow_tracking_uri": mlflow_tracking_uri, "mlflow_experiment_id": mlflow_experiment_id},
#     python_callable=train_crossval_model,
#     dag=dag,
# )

# train_resnet50_op = PythonOperator(
#     task_id="train_resnet50_model",
#     provide_context=True,
#     op_kwargs={"mlflow_tracking_uri": mlflow_tracking_uri, "mlflow_experiment_id": mlflow_experiment_id},
#     python_callable=train_resnet50_model,
#     dag=dag,
# )

compare_models_op = PythonOperator(
    task_id="compare_models",
    provide_context=True,
    op_kwargs={"mlflow_tracking_uri": mlflow_tracking_uri},
    python_callable=compare_models,
    dag=dag,
)

serve_model_op = PythonOperator(
    task_id="serve_model",
    provide_context=True,
    op_kwargs={"mlflow_tracking_uri": mlflow_tracking_uri},
    python_callable=serve_model,
    dag=dag,
)

# set task dependencies
# run_preprocessing_op >> train_basic_model_op
# run_preprocessing_op >> train_crossval_model_op
# run_preprocessing_op >> train_resnet50_op

# train_basic_model_op >> compare_models_op
# train_crossval_model_op >> compare_models_op
# train_resnet50_op >> compare_models_op

compare_models_op >> serve_model_op

# docker build --tag ${ECR}/mlflow_serving:v1 --file Dockerfile .
# docker push ${ECR}/mlflow_serving:v1
