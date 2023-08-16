import pendulum
import yaml
from airflow.decorators import dag, task
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task_sensor import ExternalTaskSensor
from kubernetes import client, config


##### AIRFLOW DAG
#
@dag(
    "cnn_skin_cancer_deployment_pipeline",
    default_args={
        "owner": "seblum",
        "depends_on_past": False,
        "start_date": pendulum.datetime(2021, 1, 1, tz="Europe/Amsterdam"),
        "tags": ["Keras CNN to classify skin cancer"],
    },
    schedule_interval=None,
    max_active_runs=1,
)
def cnn_skin_cancer_deployment():

    trigger_deploy = ExternalTaskSensor(
        task_id="external_trigger_deploy",
        external_dag_id="cnn_skin_cancer_training_pipeline",
        external_task_id="compare-models",
        # start_date=pendulum.datetime(2021, 1, 1, tz="Europe/Amsterdam"),
        # execution_delta=timedelta(hours=1),
        # timeout=3600,
    )

    def seldon_deployment_func(**kwargs):
        seldon_deployment_dict = {
            "apiVersion": "machinelearning.seldon.io/v1alpha2",
            "kind": "SeldonDeployment",
            "metadata": {"name": "mlflow", "namespace": "seldon-system"},
            "spec": {
                "protocol": "v2",
                "name": "wines",
                "predictors": [
                    {
                        "graph": {
                            "children": [],
                            "implementation": "MLFLOW_SERVER",
                            "modelUri": "s3://d7k27cmkytac-mlflow-artifact-bucket/test/elasticnet_wine_44eb4bd043964a34be556172a710bc18",
                            "name": "classifier",
                        },
                        "name": "default",
                        "replicas": 1,
                    }
                ],
            },
        }

        print("The python dictionary is:")
        print(seldon_deployment_dict)
        seldon_deployment_yaml = yaml.dump(seldon_deployment)
        print("The YAML string is:")
        print(seldon_deployment_yaml)

        print("loading config...")
        config.load_kube_config()

        k8s_apps_v1 = client.AppsV1Api()
        resp = k8s_apps_v1.create_namespaced_deployment(body=seldon_deployment_yaml, namespace="default")
        print("Deployment created. status='%s'" % resp.metadata.name)

    seldon_deployment = PythonOperator(task_id="some_task", python_callable=seldon_deployment_func)

    trigger_deploy >> seldon_deployment


cnn_skin_cancer_deployment()
