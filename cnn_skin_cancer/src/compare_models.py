import os
from typing import Tuple

import mlflow


def compare_models(input_dict: dict, metric: str = "prediction_accuracy") -> Tuple[str, str, int]:
    """
    Compares a given set of MLflow models based on their logged metric. The model with the best metric will be
    transferred to a "Staging" stage within the MLflow Registry.

    Args:
        input_dict (dict): A dictionary containing the names and run IDs of the MLflow models to compare.
        metric (str, optional): The metric to compare the models. Defaults to "prediction_accuracy".

    Returns:
        Tuple[str, str, int]: A tuple containing the name of the best performing model, its MLflow URI,
                              and the version of the model.

    Raises:
        None
    """

    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)

    all_results = {}
    for key, value in input_dict.items():
        # extract params/metrics data for run `test_run_id` in a single dict
        model_results_data_dict = client.get_run(value).data.to_dictionary()
        # get params and metrics for this run (test_run_id)
        model_results_accuracy = model_results_data_dict["metrics"][metric]
        all_results[key] = model_results_accuracy

    # Get model with maximum accuracy
    serving_model_name = max(all_results, key=all_results.get)
    serving_model_version = client.get_latest_versions(name=serving_model_name, stages=["None"])[0].version
    print(f"acc_dict: {all_results}")
    print(f"acc_dict_model: {serving_model_name}")
    print(f"latest_model_version: {serving_model_version}")

    # Transition model to stage "Staging"
    model_stage = "Staging"
    client.transition_model_version_stage(name=serving_model_name, version=serving_model_version, stage=model_stage)
    serving_model_uri = f"models:/{serving_model_name}/{model_stage}"

    return serving_model_name, serving_model_uri, serving_model_version


if __name__ == "__main__":
    input_dict = {}
    compare_models(input_dict=input_dict)
