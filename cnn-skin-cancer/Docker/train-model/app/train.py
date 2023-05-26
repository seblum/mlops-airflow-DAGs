import mlflow
from keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50

# ----- ----- ----- ----- ----- -----
## RESNET 50

params = {
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
}

model = get_model()
RUN_NAME = "resnet50-cnn"


def train_model(mlflow_tracking_uri: str, mlflow_experiment_id: str, model, params: dict, run_name: str, **kwargs):
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    ti = kwargs["ti"]

    path_X_train = ti.xcom_pull(key="path_X_train", task_ids="run_preprocessing")
    path_y_train = ti.xcom_pull(key="path_y_train", task_ids="run_preprocessing")
    path_X_test = ti.xcom_pull(key="path_X_test", task_ids="run_preprocessing")
    path_y_test = ti.xcom_pull(key="path_y_test", task_ids="run_preprocessing")

    # TODO: replace with s3 load
    X_train = np.load(f"{path_X_train}")
    y_train = np.load(f"{path_y_train}")
    X_test = np.load(f"{path_X_test}")
    y_test = np.load(f"{path_y_test}")

    learning_rate_reduction = ReduceLROnPlateau(monitor="accuracy", patience=5, verbose=1, factor=0.5, min_lr=1e-7)

    with mlflow.start_run(experiment_id=mlflow_experiment_id, run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.log_params(params)

        mlflow.keras.autolog()
        # Train ResNet50 on all the data
        model.fit(
            X_train,
            y_train,
            validation_split=params.get("validation_split"),
            epochs=params.get("epochs"),
            batch_size=params.get("batch_size"),
            verbose=2,
            callbacks=[learning_rate_reduction],
        )
        # Train ResNet50 on all the data
        mlflow.keras.autolog(disable=True)
        model_uri = f"runs:/{run_id}/{run_name}"

        # Testing model on test data to evaluate
        y_pred = model.predict(X_test)
        prediction_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        mlflow.log_metric("prediction_accuracy", prediction_accuracy)
        print(prediction_accuracy)

        mlflow.keras.log_model(model, artifact_path=run_name)

        mv = mlflow.register_model(model_uri, run_name)
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))
        print("Stage: {}".format(mv.current_stage))

    kwargs["ti"].xcom_push(key=f"run_id-{run_name}", value=run_id)
    kwargs["ti"].xcom_push(key=f"model_version-{run_name}", value=mv.version)
