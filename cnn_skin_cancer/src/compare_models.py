import os
from pprint import pprint

import mlflow


def compare_models(mlflow_tracking_uri: str, input_dict: dict, **kwargs) -> None:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)

    # TODO: Loop over input dict, save all parameters in tmp dict, return max model
    all_results = {}
    for key, value in input_dict.items():
        # extract params/metrics data for run `test_run_id` in a single dict
        model_results_data_dict = client.get_run(value).data.to_dictionary()
        # get params and metrics for this run (test_run_id)
        model_results_accuracy = model_results_data_dict["metrics"]["prediction_accuracy"]
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
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    compare_models(mlflow_tracking_uri=mlflow_tracking_uri)
