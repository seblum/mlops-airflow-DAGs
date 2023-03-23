import mlflow
from pprint import pprint


def serve_model(mlflow_tracking_uri: str, **kwargs) -> str:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    ti = kwargs["ti"]

    client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)

    run_id_basic = ti.xcom_pull(key="run_id-basic-keras-cnn", task_ids="train_basic_model")
    run_id_crossval = ti.xcom_pull(key="run_id-crossval-keras-cnn", task_ids="train_crossval_model")
    run_id_resnet50 = ti.xcom_pull(key="run_id-resnet50-cnn", task_ids="train_resnet50_model")

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
    print(acc_dict)

    # get model by maximum accuracy
    acc_dict_model = max(acc_dict, key=acc_dict.get)
    print(f"Maximal Accuracy is: acc_dict_model ({acc_dict.get(acc_dict_model)})")
    # set env to production
    client.transition_model_version_stage(name=acc_dict_model, staging="Production")

    # client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{acc_dict_model}/Production")
    print(model)
