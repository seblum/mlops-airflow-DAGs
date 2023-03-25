import mlflow
from pprint import pprint


def serve_model(mlflow_tracking_uri:str,**kwargs) -> str:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        client = mlflow.MlflowClient(tracking_uri=mlflow_tracking_uri)

        ti = kwargs['ti']
        model_stage = "Staging"

        serving_model_name = ti.xcom_pull(key="serving_model_name", task_ids='compare_models')
        serving_model_uri = ti.xcom_pull(key="serving_model_uri", task_ids='compare_models')
        latest_model_version = client.get_latest_versions(name = serving_model_name, stages = [{model_stage}])[0].version
        print(latest_model_version)

        model = mlflow.pyfunc.load_model(model_uri=serving_model_uri)
        print(model.version)
        print(latest_model_version)
        print(model.stage)

        # build docker image

        # buildimp
        # mlflow models serve -m "models:/sk-learn-random-forest-reg-model/Production"

        # https://www.lucidchart.com/techblog/2019/03/22/using-apache-airflows-docker-operator-with-amazons-container-repository/