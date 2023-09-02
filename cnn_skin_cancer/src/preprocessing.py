import os
from datetime import datetime
from typing import Tuple

import mlflow
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle
from src.utils import AWSSession, timeit
from tqdm import tqdm


@timeit
def data_preprocessing(
    mlflow_experiment_id: str,
    aws_bucket: str,
    path_preprocessed: str = "preprocessed",
) -> Tuple[str, str, str, str]:
    """Preprocesses data for further use within model training. Raw data is read from given S3 Bucket, normalized, and stored ad a NumPy Array within S3 again. Output directory is on "/preprocessed". The shape of the data set is logged to MLflow.

    Args:
        mlflow_experiment_id (str): Experiment ID of the MLflow run to log data
        aws_bucket (str): S3 Bucket to read raw data from and write preprocessed data
        path_preprocessed (str, optional): Subdirectory to store the preprocessed data on the provided S3 Bucket. Defaults to "preprocessed".

    Returns:
        Tuple[str, str, str, str]: Four strings denoting the path of the preprocessed data stored as NumPy Arrays: X_train_data_path, y_train_data_path, X_test_data_path, y_test_data_path
    """
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Instantiate aws session based on AWS Access Key
    # AWS Access Key is fetched within AWS Session by os.getenv
    aws_session = AWSSession()
    aws_session.set_sessions()

    # Set paths within s3
    path_raw_data = f"s3://{aws_bucket}/data/"

    folder_benign_train = f"{path_raw_data}train/benign"
    folder_malignant_train = f"{path_raw_data}train/malignant"

    folder_benign_test = f"{path_raw_data}test/benign"
    folder_malignant_test = f"{path_raw_data}test/malignant"

    @timeit
    def _load_and_convert_images(folder_path: str) -> np.array:
        """
        Loads and converts images from an S3 bucket folder into a NumPy array.

        Args:
            folder_path (str): The path to the S3 bucket folder.

        Returns:
            np.array: The NumPy array containing the converted images.

        Raises:
            None
        """
        ims = [
            aws_session.read_image_from_s3(s3_bucket=aws_bucket, imname=filename)
            # TODO: currently only uses the last ten files for testing
            # for filename in tqdm(aws_session.list_files_in_bucket(folder_path)[-10:])
            for filename in tqdm(aws_session.list_files_in_bucket(folder_path))
        ]
        return np.array(ims, dtype="uint8")

    def _create_label(x_dataset: np.array) -> np.array:
        """
        Creates label array for the given dataset.

        Args:
            x_dataset (np.array): The dataset for which labels are to be created.

        Returns:
            np.array: The label array.

        Raises:
            None
        """
        return np.zeros(x_dataset.shape[0])

    def _merge_data(set_one: np.array, set_two: np.array) -> np.array:
        """
        Merges two datasets into a single dataset.

        Args:
            set_one (np.array): The first dataset.
            set_two (np.array): The second dataset.

        Returns:
            np.array: The merged dataset.

        Raises:
            None
        """
        return np.concatenate((set_one, set_two), axis=0)

    # Start a MLflow run to log the size of the data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(experiment_id=mlflow_experiment_id, run_name=f"{timestamp}_Preprocessing") as run:
        print("\n> Loading images from S3...")
        # Load in training pictures
        X_benign = _load_and_convert_images(folder_benign_train)
        X_malignant = _load_and_convert_images(folder_malignant_train)

        # Load in testing pictures
        X_benign_test = _load_and_convert_images(folder_benign_test)
        X_malignant_test = _load_and_convert_images(folder_malignant_test)

        # Log train-test size in MLflow
        print("\n> Log data parameters")
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

    X_train_data_path = f"{path_preprocessed}/X_train.pkl"
    y_train_data_path = f"{path_preprocessed}/y_train.pkl"
    X_test_data_path = f"{path_preprocessed}/X_test.pkl"
    y_test_data_path = f"{path_preprocessed}/y_test.pkl"

    return X_train_data_path, y_train_data_path, X_test_data_path, y_test_data_path


if __name__ == "__main__":
    mlflow_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    aws_bucket = os.getenv("AWS_BUCKET")

    data_preprocessing(
        mlflow_experiment_id=mlflow_experiment_id,
        aws_bucket=aws_bucket,
    )
