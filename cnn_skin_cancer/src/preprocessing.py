import json
import os
from datetime import datetime

import mlflow
import numpy as np
from keras.utils.np_utils import (  # used for converting labels to one-hot-encoding
    to_categorical,
)
from sklearn.utils import shuffle
from tqdm import tqdm
from utils import AWSSession, timeit


@timeit
def data_preprocessing(
    mlflow_tracking_uri: str,
    mlflow_experiment_id: str,
    aws_bucket: str,
    path_preprocessed: str = "preprocessed",
    **kwargs,
) -> json:
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # read_image = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_ROLE_NAME = os.getenv("AWS_ROLE_NAME")
    AWS_REGION = os.getenv("AWS_REGION")

    # need to check that I instatiate this within airflow dags with correct access key
    aws_session = AWSSession(
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_role_name=AWS_ROLE_NAME,
    ).set_sessions()

    # Set paths within s3
    path_raw_data = f"s3://{aws_bucket}/data/"

    folder_benign_train = f"{path_raw_data}train/benign"
    folder_malignant_train = f"{path_raw_data}train/malignant"

    folder_benign_test = f"{path_raw_data}test/benign"
    folder_malignant_test = f"{path_raw_data}test/malignant"

    # Define Processing methods
    @timeit
    def _load_and_convert_images(folder_path: str) -> np.array:
        ims = [
            aws_session.read_image_from_s3(s3_bucket=aws_bucket, imname=filename)
            for filename in tqdm(aws_session.list_files_in_bucket(folder_path)[-10:])
        ]
        return np.array(ims, dtype="uint8")

    def _create_label(x_dataset: np.array) -> np.array:
        return np.zeros(x_dataset.shape[0])

    def _merge_data(set_one: np.array, set_two: np.array) -> np.array:
        return np.concatenate((set_one, set_two), axis=0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(experiment_id=mlflow_experiment_id, run_name=f"{timestamp}_Preprocessing") as run:
        print("\n> Loading images from S3...")
        # Load in training pictures
        X_benign = _load_and_convert_images(folder_benign_train)
        X_malignant = _load_and_convert_images(folder_malignant_train)

        # Load in testing pictures
        X_benign_test = _load_and_convert_images(folder_benign_test)
        X_malignant_test = _load_and_convert_images(folder_malignant_test)

        print("\n> Log data parameters")
        # Log train-test size in MLflow
        mlflow.log_param("train_size_benign", X_benign.shape[0])
        mlflow.log_param("train_size_malignant", X_malignant.shape[0])
        mlflow.log_param("test_size_benign", X_benign_test.shape[0])
        mlflow.log_param("test_size_malignant", X_malignant_test.shape[0])

        print("\n> Preprocessing...")
        # Create labels
        y_benign = _create_label(X_benign)
        y_malignant = _create_label(X_malignant)

        y_benign_test = _create_label(X_benign_test)
        y_malignant_test = _create_label(X_malignant_test)

        # Merge data
        y_train = _merge_data(y_benign, y_malignant)
        y_test = _merge_data(y_benign_test, y_malignant_test)

        X_train = _merge_data(X_benign, X_malignant)
        X_test = _merge_data(X_benign_test, X_malignant_test)

        # Shuffle data
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)

        # With data augmentation to prevent overfitting
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        print("\n> Upload numpy arrays to S3...")
        aws_session.upload_npy_to_s3(
            data=X_train,
            s3_bucket=aws_bucket,
            file_key=f"{path_preprocessed}/X_train.pkl",
        )
        aws_session.upload_npy_to_s3(
            data=y_train,
            s3_bucket=aws_bucket,
            file_key=f"{path_preprocessed}/y_train.pkl",
        )
        aws_session.upload_npy_to_s3(
            data=X_test,
            s3_bucket=aws_bucket,
            file_key=f"{path_preprocessed}/X_test.pkl",
        )
        aws_session.upload_npy_to_s3(
            data=y_test,
            s3_bucket=aws_bucket,
            file_key=f"{path_preprocessed}/y_test.pkl",
        )

    # Create dictionary with S3 paths to return
    # return_dict = {
    #     "X_train_data_path": f"{path_preprocessed}/X_train.pkl",
    #     "y_train_data_path": f"{path_preprocessed}/y_train.pkl",
    #     "X_test_data_path": f"{path_preprocessed}/X_test.pkl",
    #     "y_test_data_path": f"{path_preprocessed}/y_test.pkl",
    # }
    # return json.dumps(return_dict)

    X_train_data_path = f"{path_preprocessed}/X_train.pkl"
    y_train_data_path = f"{path_preprocessed}/y_train.pkl"
    X_test_data_path = f"{path_preprocessed}/X_test.pkl"
    y_test_data_path = f"{path_preprocessed}/y_test.pkl"

    return X_train_data_path, y_train_data_path, X_test_data_path, y_test_data_path

    # return "hello"
    # ti.xcom_push(key="X_train_data_path", value=f"{path_preprocessed}/X_train.pkl")
    # ti.xcom_push(key="y_train_data_path", value=f"{path_preprocessed}/y_train.pkl")
    # ti.xcom_push(key="X_test_data_path", value=f"{path_preprocessed}/X_test.pkl")
    # ti.xcom_push(key="y_test_data_path", value=f"{path_preprocessed}/y_test.pkl")


if __name__ == "__main__":
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    aws_bucket = os.getenv("AWS_BUCKET")

    data_preprocessing(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_id=mlflow_experiment_id,
        aws_bucket=aws_bucket,
    )
