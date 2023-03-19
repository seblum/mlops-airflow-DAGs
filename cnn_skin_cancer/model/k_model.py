from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau

import mlflow
import mlflow.keras
import numpy as np

def run_model(mlflow_run_id, X_train, y_train, X_test, y_test, **kwargs):

    ti = kwargs['ti']

    # get value_1
    path_X_train = ti.xcom_pull(key="path_X_train", task_ids='run_preprocessing')

    X_train = np.load(f'{path_X_train}')

    params = {
        "num_classes": 2,
        "input_shape": (224, 224, 3),
        "activation": "relu",
        "kernel_initializer": "glorot_uniform",
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy"],
        "validation_split": 0.2,
        "epochs": 10,
        "batch_size": 64,
    }

    model = Sequential(
        [
            # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(
                64,
                kernel_size=(3, 3),
                padding="Same",
                input_shape=params.get("input_shape"),
                activation=params.get("activation"),
                kernel_initializer=params.get("kernel_initializer"),
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(
                64,
                kernel_size=(3, 3),
                padding="Same",
                activation=params.get("activation"),
                kernel_initializer=params.get("kernel_initializer"),
            ),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation=params.get("activation"), kernel_initializer="normal"),
            layers.Dense(params.get("num_classes"), activation="softmax"),
        ]
    )

    model.compile(optimizer=params.get("optimizer"), loss=params.get("loss"), metrics=params.get("metrics"))
    model.summary()

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=5, verbose=1, factor=0.5, min_lr=1e-7)

    # learning_rate
    lr = 1e-5

    with mlflow.start_run(run_id=mlflow_run_id) as run:

        mlflow.log_params(params)
        # mlflow.set_tag("env", "dev")

        mlflow.keras.autolog()
        history = model.fit(
            X_train,
            y_train,
            validation_split=params.get("validation_split"),
            epochs=params.get("epochs"),
            batch_size=params.get("batch_size"),
            verbose=1,
            callbacks=[learning_rate_reduction],
        )

        mlflow.keras.autolog(disable=True)
        mlflow.keras.log_model(model, artifact_path="keras-model")

        # list all data in history
        print(history.history.keys())
        # summarize history for accuracy
        fig = plt.figure(figsize=(3, 6))
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        mlflow.log_figure(fig, "accuracy.png")

        # summarize history for loss
        fig = plt.figure(figsize=(3, 6))
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        mlflow.log_figure(fig, "loss.png")

        # Testing model on test data to evaluate
        y_pred = model.predict(X_test)

        # keras_model = mlflow.keras.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2" + "/models")
        # predictions = keras_model.predict(x_test)

        print(accuracy_score(np.argmax(y_test, axis=1), y_pred))
