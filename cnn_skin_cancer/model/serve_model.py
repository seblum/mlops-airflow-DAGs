import mlflow
from pprint import pprint

def compare_models(mlflow_tracking_uri:str,mlflow_experiment_id:str, **kwargs):

    client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)
    # for rm in client.search_registered_models():
    #     pprint(dict(rm), indent=4)
    
    # get model names

    # list all experiment at this Tracking server
    client.list_experiments()

    ti = kwargs['ti']
    run_id_basic = ti.xcom_pull(key="run_id-basic-keras-cnn", task_ids='train_basic_model')
    run_id_crossval = ti.xcom_pull(key="run_id-crossval-keras-cnn", task_ids='train_crossval_model')
    run_id_resnet50 = ti.xcom_pull(key="run_id-resnet50-cnn", task_ids='train_resnet50_model')

    # extract params/metrics data for run `test_run_id` in a single dict 
    basic_data_basic = client.get_run(run_id_basic).data.to_dictionary()
    crossval_data_dict = client.get_run(run_id_crossval).data.to_dictionary()
    resnet50_data_dict = client.get_run(run_id_resnet50).data.to_dictionary()

    # list all params and metrics for this run (test_run_id)
    # pprint(run_data_dict)
    acc_dict = {"basic-keras-cnn": basic_data_basic['metrics']['prediction_accuracy'],
            "crossval-keras-cnn":crossval_data_dict['metrics']['prediction_accuracy'],
            "resnet50-cnn":resnet50_data_dict['metrics']['prediction_accuracy']
    }
    # get metric from each model
    #  metrics = client.get_metric_history(runID, metricKey)

    acc_dict_model = max(acc_dict, key=acc_dict.get)

    # set env to production
    client.transition_model_version_stage(
    name=acc_dict_model, staging="Production"
    )
    # serve model



def serve_model():
    pass