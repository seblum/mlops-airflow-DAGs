import pendulum
from airflow.decorators import dag, task
from airflow.kubernetes.secret import Secret
from airflow.models import Variable

################################################################################
#
# SET VARIOUS PARAMETERS
#
skin_cancer_container_image = "seblum/cnn-skin-cancer-model:latest"  # base image for k8s pods

# EXPERIMENT_NAME = "cnn_skin_cancer"  # mlflow experiment name
# MLFLOW_TRACKING_URI = Variable.get("MLFLOW_TRACKING_URI")
# ECR_REPOSITORY_NAME = Variable.get("ECR_REPOSITORY_NAME")
# ECR_SAGEMAKER_IMAGE_TAG = Variable.get("ECR_SAGEMAKER_IMAGE_TAG")


secret_name = "airflow-aws-account-information"
# SECRET_AWS_ID = Secret(deploy_type="env", deploy_target="AWS_ID", secret=secret_name, key="AWS_ID")
SECRET_AWS_REGION = Secret(deploy_type="env", deploy_target="AWS_REGION", secret=secret_name, key="AWS_REGION")


################################################################################
#
# AIRFLOW DAG
#
@dag(
    dag_id="cnn_skin_cancer_sagemaker_test_inference",
    default_args={
        "owner": "seblum",
        "depends_on_past": False,
        "start_date": pendulum.datetime(2021, 1, 1, tz="Europe/Amsterdam"),
        "tags": ["Inference test on CNN sagemaker deployment"],
    },
    schedule_interval=None,
    max_active_runs=1,
)
def cnn_skin_cancer_sagemaker_inference_test():
    @task.kubernetes(
        image=skin_cancer_container_image,
        task_id="inference_call_op",
        namespace="airflow",
        # env_vars={"MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
        in_cluster=True,
        get_logs=True,
        # do_xcom_push=True,
        startup_timeout_seconds=300,
        service_account_name="airflow-sa",
        secrets=[
            SECRET_AWS_REGION,
        ],
    )
    def inference_call_op():

        sagemaker_endpoint_name = "test-cnn-skin-cancer"
        region = "eu-central-1"

        import json

        from src.inference_to_sagemaker import (
            check_status,
            get_image_directory,
            preprocess_image,
            query_endpoint,
            read_imagefile,
        )

        filenames = ["1.jpg", "10.jpg", "1003.jpg", "1005.jpg", "1007.jpg"]

        path = get_image_directory()
        print(path)

        file = f"{path}/1.jpg"

        # Check endpoint status
        print("[+] Endpoint Status")
        print(f"Application status is {check_status(sagemaker_endpoint_name)}")

        image = read_imagefile(file)

        print("[+] Preprocess Data")
        np_image = preprocess_image(image)

        # payload = json.dumps(np_image.tolist())

        payload = json.dumps({"instances": np_image.tolist()})

        print("[+] Prediction")
        predictions = query_endpoint(app_name=sagemaker_endpoint_name, data=payload)
        print(predictions)

    inference_call_op()


cnn_skin_cancer_sagemaker_inference_test()
