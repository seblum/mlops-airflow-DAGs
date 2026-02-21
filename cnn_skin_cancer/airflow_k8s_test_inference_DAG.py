"""
Inference test DAG: validates a deployed SageMaker endpoint using sample images.

AWS_REGION is resolved from the Kubernetes Secrets Backend (K8s secret synced
by External Secrets Operator) — no explicit Secret() objects needed.
"""

import pendulum
from airflow.decorators import dag, task

skin_cancer_container_image = "seblum/cnn-skin-cancer-model:latest"


@dag(
    dag_id="cnn_skin_cancer_sagemaker_test_inference",
    default_args={
        "owner": "seblum",
        "depends_on_past": False,
        "start_date": pendulum.datetime(2021, 1, 1, tz="Europe/Amsterdam"),
    },
    tags=["Inference test on CNN sagemaker deployment"],
    schedule=None,
    max_active_runs=1,
)
def cnn_skin_cancer_sagemaker_inference_test():
    """Apache Airflow DAG for testing inference on a CNN SageMaker deployment."""

    @task.kubernetes(
        image=skin_cancer_container_image,
        task_id="inference_call_op",
        namespace="airflow",
        in_cluster=True,
        get_logs=True,
        startup_timeout_seconds=300,
        service_account_name="airflow-sa",
        # AWS_REGION resolved from K8s secret via Kubernetes Secrets Backend
    )
    def inference_call_op() -> None:
        """Run inference against the SageMaker endpoint for a set of test images."""
        import json

        from src.inference_to_sagemaker import (
            endpoint_status,
            get_image_directory,
            preprocess_image,
            query_endpoint,
            read_imagefile,
        )

        sagemaker_endpoint_name = "test-cnn-skin-cancer"
        image_directory = get_image_directory()
        filenames = ["1.jpg", "10.jpg", "1003.jpg", "1005.jpg", "1007.jpg"]

        for filename in filenames:
            filepath = f"{image_directory}/{filename}"
            print(f"[+] Running inference on {filepath}")
            print(f"[+] Endpoint status: {endpoint_status(sagemaker_endpoint_name)}")

            image = read_imagefile(filepath)
            np_image = preprocess_image(image)
            payload = json.dumps({"instances": np_image.tolist()})
            predictions = query_endpoint(app_name=sagemaker_endpoint_name, data=payload)
            print(f"[+] Predictions for {filename}: {predictions}")

    inference_call_op()


cnn_skin_cancer_sagemaker_inference_test()
