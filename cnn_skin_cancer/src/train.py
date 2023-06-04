import json
import os
from datetime import datetime
from enum import Enum

import mlflow
import mlflow.keras
import numpy as np
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau

# from keras.optimizers import Adam, RMSprop
from model.utils import Model_Class, get_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tensorflow.keras.applications.resnet50 import ResNet50
from utils import AWSSession


def train_model(
    mlflow_tracking_uri: str,
    mlflow_experiment_id: str,
    model_class: Enum,
    model_params: dict,
    aws_bucket: str,
    import_dict: dict = {},
    **kwargs,
):
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    # ti = kwargs["ti"]
    print("\n> Loading data...")

    X_train_data_path = import_dict.get("X_train_data_path")
    y_train_data_path = import_dict.get("y_train_data_path")
    X_test_data_path = import_dict.get("X_test_data_path")
    y_test_data_path = import_dict.get("y_test_data_path")
    print(y_train_data_path)
    # # X_train_data_path = ti.xcom_pull(key="X_train_data_path", task_ids="run_preprocessing")
    # # y_train_data_path = ti.xcom_pull(key="y_train_data_path", task_ids="run_preprocessing")
    # # X_test_data_path = ti.xcom_pull(key="X_test_data_path", task_ids="run_preprocessing")
    # # y_test_data_path = ti.xcom_pull(key="y_test_data_path", task_ids="run_preprocessing")

    # # for local testing
    # path_preprocessed = "preprocessed"
    # X_train_data_path = f"{path_preprocessed}/X_train.pkl"
    # y_train_data_path = f"{path_preprocessed}/y_train.pkl"
    # X_test_data_path = f"{path_preprocessed}/X_test.pkl"
    # y_test_data_path = f"{path_preprocessed}/y_test.pkl"

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
    )
    aws_session.set_sessions()

    X_train = aws_session.download_npy_from_s3(s3_bucket=aws_bucket, file_key=X_train_data_path)
    y_train = aws_session.download_npy_from_s3(s3_bucket=aws_bucket, file_key=y_train_data_path)
    X_test = aws_session.download_npy_from_s3(s3_bucket=aws_bucket, file_key=X_test_data_path)
    y_test = aws_session.download_npy_from_s3(s3_bucket=aws_bucket, file_key=y_test_data_path)

    print("\n> Training model...")
    run_name = model_class
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mlflow.start_run(experiment_id=mlflow_experiment_id, run_name=f"{timestamp}-{run_name}") as run:
        mlflow.log_params(model_params)
        learning_rate_reduction = ReduceLROnPlateau(monitor="accuracy", patience=5, verbose=1, factor=0.5, min_lr=1e-7)

        if model_class == Model_Class.CrossVal:
            kfold = KFold(n_splits=3, shuffle=True, random_state=11)
            cvscores = []
            for train, test in kfold.split(X_train, y_train):
                model = get_model(Model_Class.Basic, model_params)
                # TODO: autolog kfold ???
                model.fit(
                    X_train[train],
                    y_train[train],
                    epochs=model_params.get("epochs"),
                    batch_size=model_params.get("batch_size"),
                    verbose=model_params.get("verbose"),
                )
                scores = model.evaluate(X_train[test], y_train[test], verbose=0)
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
                cvscores.append(scores[1] * 100)
                K.clear_session()
        else:
            model = get_model(model_class, model_params)
            mlflow.keras.autolog()
            model.fit(
                X_train,
                y_train,
                validation_split=model_params.get("validation_split"),
                epochs=model_params.get("epochs"),
                batch_size=model_params.get("batch_size"),
                verbose=model_params.get("verbose"),
                callbacks=[learning_rate_reduction],
            )
            mlflow.keras.autolog(disable=True)

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/{run_name}"

        # Testing model on test data to evaluate
        print("\n> Testing model...")
        y_pred = model.predict(X_test)
        prediction_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        mlflow.log_metric("prediction_accuracy", prediction_accuracy)
        print(f"Prediction Accuracy: {prediction_accuracy}")

        # mlflow.keras.log_model(model, artifact_path=run_name)

        print("\n> Register model")
        mv = mlflow.register_model(model_uri, run_name)
        print(f"Name: {mv.name}")
        print(f"Version: {mv.version}")
        print(f"Stage: {mv.current_stage}")

    # # Create dictionary with S3 paths to return
    # return_dict = {
    #     f"run_id-{run_name}": run_id,
    #     f"model_version-{run_name}": mv.version
    # }
    # return json.dumps(return_dict)

    # Create dictionary with S3 paths to return

    return run_id, mv.version, mv.name, mv.current_stage

    # print(f"run_id-{run_name}" run_id)
    # print(f"model_version-{run_name}" mv.version)
    # kwargs["ti"].xcom_push(key=f"run_id-{run_name}", value=run_id)
    # kwargs["ti"].xcom_push(key=f"model_version-{run_name}", value=mv.version)


if __name__ == "__main__":
    mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID")
    aws_bucket = os.getenv("AWS_BUCKET")
    model_class = os.getenv("MODEL_CLASS")

    # deleted afterwards
    model_params = {
        "num_classes": 2,
        "input_shape": (224, 224, 3),
        "activation": "relu",
        "kernel_initializer_glob": "glorot_uniform",
        "kernel_initializer_norm": "normal",
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy"],
        "validation_split": 0.2,
        "epochs": 2,
        "batch_size": 64,
        "learning_rate": 1e-5,
        "pooling": "avg",  # needed for resnet50
        "verbose": 2,
    }

    train_model(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_id=mlflow_experiment_id,
        model_class=Model_Class[model_class],
        model_params=model_params,
        aws_bucket=aws_bucket,
    )
