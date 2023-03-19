from cnn_skin_cancer.model.preprocessing import run_preprocessing
from cnn_skin_cancer.model.k_model import run_model

import pendulum
import mlflow

from airflow import DAG
from airflow.operators.python import PythonOperator


mlflow.set_tracking_uri("http://127.0.0.1:5006/")

experiment_name = "tracking-experiment"
run_name = "tracking-example-run"

try:
    # Creating an experiment
    mlflow.create_experiment(experiment_name)
except:
    pass
# Setting the environment with the created experiment
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=run_name) as run:
    mlflow_run_id = run.info.run_id


default_args = {
    "owner": "seblum",
    "depends_on_past": False,
    "start_date": pendulum.datetime(2021, 1, 1, tz="UTC"),
    "provide_context": True,
    "tags": ["Keras CNN to classify skin cancer"],
}
dag = DAG("skin_cancer_cnn", default_args=default_args, schedule_interval=None, max_active_runs=1)

# KubernetesPodOperator
run_preprocessing_op = PythonOperator(
    task_id="run_preprocessing",
    provide_context=True,
    python_callable=run_preprocessing,
    op_kwargs={"mlflow_run_id": mlflow_run_id},
    dag=dag,
)

# run_model_kwargs = {
#     "mlflow_run_id": mlflow_run_id,
#     "X_train": X_train,
#     "y_train": y_train,
#     "X_test": X_test,
#     "y_test": y_test,
# }
run_model_op = PythonOperator(
    task_id="run_model", provide_context=True, python_callable=run_model, dag=dag
)


# set task dependencies
run_preprocessing_op >> run_model_op


# X_train, y_train, X_test, y_test = run_preprocessing(mlflow_run_id)

# run_model(mlflow_run_id, X_train, y_train, X_test, y_test)
