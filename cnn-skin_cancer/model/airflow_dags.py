from preprocessing import run_preprocessing
from k_model import run_model

from airflow import DAG

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5005/")


run_name = "tracking-example-run"
experiment_name = "tracking-experiment"

try:
    # Creating an experiment
    mlflow.create_experiment(experiment_name)
except:
    pass
# Setting the environment with the created experiment
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=run_name) as run:
    mlflow_run_id = run.info.run_id


X_train, y_train, X_test, y_test = run_preprocessing(mlflow_run_id)

run_model(mlflow_run_id, X_train, y_train, X_test, y_test)
