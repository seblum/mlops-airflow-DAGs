import os
from pprint import pprint

import mlflow


def compare_models(mlflow_tracking_uri: str, **kwargs) -> None:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    ti = kwargs["ti"]

    client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)

    # used for testing
    # can be deleted afterward
    kwargs["ti"].xcom_push(key=f"run_id-basic-keras-cnn", value="c7cdeb90280f4877adddd1656de8b606")
    kwargs["ti"].xcom_push(key=f"run_id-crossval-keras-cnn", value="b441fb2eaa4a43919e9377e71404a0e8")
    kwargs["ti"].xcom_push(key=f"run_id-resnet50-cnn", value="a5cdf16cb8524a8a9168316f20772e27")

    run_id_basic = ti.xcom_pull(key="run_id-basic-keras-cnn", task_ids="compare_models")
    run_id_crossval = ti.xcom_pull(key="run_id-crossval-keras-cnn", task_ids="compare_models")
    run_id_resnet50 = ti.xcom_pull(key="run_id-resnet50-cnn", task_ids="compare_models")

    # run_id_basic = ti.xcom_pull(key="run_id-basic-keras-cnn", task_ids='train_basic_model')
    # run_id_crossval = ti.xcom_pull(key="run_id-crossval-keras-cnn", task_ids='train_crossval_model')
    # run_id_resnet50 = ti.xcom_pull(key="run_id-resnet50-cnn", task_ids='train_resnet50_model')

    # extract params/metrics data for run `test_run_id` in a single dict
    basic_data_basic = client.get_run(run_id_basic).data.to_dictionary()
    crossval_data_dict = client.get_run(run_id_crossval).data.to_dictionary()
    resnet50_data_dict = client.get_run(run_id_resnet50).data.to_dictionary()

    # list all params and metrics for this run (test_run_id)
    # get metric from each model
    acc_dict = {
        "basic-keras-cnn": basic_data_basic["metrics"]["prediction_accuracy"],
        "crossval-keras-cnn": crossval_data_dict["metrics"]["prediction_accuracy"],
        "resnet50-cnn": resnet50_data_dict["metrics"]["prediction_accuracy"],
    }

    # Get model with maximum accuracy
    acc_dict_model = max(acc_dict, key=acc_dict.get)
    latest_model_version = client.get_latest_versions(name=acc_dict_model, stages=["None"])[0].version
    print(f"acc_dict: {acc_dict}")
    print(f"acc_dict_model: {acc_dict_model}")
    print(f"latest_model_version: {latest_model_version}")

    # Transition model to stage "Staging"
    model_stage = "Staging"
    client.transition_model_version_stage(name=acc_dict_model, version=latest_model_version, stage={model_stage})
    model_uri = f"models:/{acc_dict_model}/{model_stage}"

    kwargs["ti"].xcom_push(key="serving_model_name", value=acc_dict_model)
    kwargs["ti"].xcom_push(key="serving_model_uri", value=model_uri)


if __name__ == "__main__":
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    compare_models(mlflow_tracking_uri=mlflow_tracking_uri)
