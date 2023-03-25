import mlflow
from keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.model_selection import KFold
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score

# ----- ----- ----- ----- ----- -----
# CROSS VALIDATION

K.clear_session()

def train_crossval_model(mlflow_tracking_uri:str,mlflow_experiment_id:str, **kwargs):
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    ti = kwargs['ti']

    path_X_train = ti.xcom_pull(key="path_X_train", task_ids='run_preprocessing')
    path_y_train = ti.xcom_pull(key="path_y_train", task_ids='run_preprocessing')
    path_X_test = ti.xcom_pull(key="path_X_test", task_ids='run_preprocessing')
    path_y_test = ti.xcom_pull(key="path_y_test", task_ids='run_preprocessing')

    X_train = np.load(f'{path_X_train}')
    y_train = np.load(f'{path_y_train}')
    X_test = np.load(f'{path_X_test}')
    y_test = np.load(f'{path_y_test}')

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
        "lr": 1e-5,
    }

    run_name = "crossval-keras-cnn"
    with mlflow.start_run(experiment_id=mlflow_experiment_id,run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.log_params(params)

        # define 3-fold cross validation test harness
        kfold = KFold(n_splits=3, shuffle=True, random_state=11)

        cvscores = []
        for train, test in kfold.split(X_train, y_train):
        # create model
    
            model = Sequential(
                [
                    # layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
                    layers.Conv2D(
                        64,
                        kernel_size=(3, 3),
                        padding="Same",
                        input_shape=params.get("input_shape"),
                        activation=params.get("activation"),
                        kernel_initializer=params.get("kernel_initializer_glob"),
                    ),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Dropout(0.25),
                    layers.Conv2D(
                        64,
                        kernel_size=(3, 3),
                        padding="Same",
                        activation=params.get("activation"),
                        kernel_initializer=params.get("kernel_initializer_glob"),
                    ),
                    layers.MaxPooling2D(pool_size=(2, 2)),
                    layers.Dropout(0.25),
                    layers.Flatten(),
                    layers.Dense(128, activation=params.get("activation"), kernel_initializer=params.get("kernel_initializer_norm")),
                    layers.Dense(params.get("num_classes"), activation="softmax"),
                ]
            )

            model.compile(optimizer=params.get("optimizer"), loss=params.get("loss"), metrics=params.get("metrics"))
            # Fit the model
            model.fit(X_train[train], y_train[train], epochs=params.get("epochs"), batch_size=params.get("batch_size"), verbose=0)

            # evaluate the model
            scores = model.evaluate(X_train[test], y_train[test], verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            K.clear_session()

        model_uri = f"runs:/{run_id}/{run_name}"

        mlflow.keras.log_model(model, artifact_path=run_name)

        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

        # Testing model on test data to evaluate
        y_pred = model.predict(X_test)
        prediction_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        mlflow.log_metric("prediction_accuracy",prediction_accuracy)
        print(prediction_accuracy)

        mlflow.keras.log_model(model, artifact_path=run_name)

        mv = mlflow.register_model(model_uri, run_name)
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))
        print("Stage: {}".format(mv.current_stage))

    kwargs["ti"].xcom_push(key=f"run_id-{run_name}", value=run_id)
