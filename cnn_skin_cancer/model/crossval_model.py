import mlflow
from keras.optimizers import Adam, RMSprop
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.model_selection import KFold

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
        "kernel_initializer": "glorot_uniform",
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy"],
        "validation_split": 0.2,
        "epochs": 2,
        "batch_size": 64,
    }

    run_name = "basic-keras-cnn"
    with mlflow.start_run(experiment_id=mlflow_experiment_id,run_name=run_name) as run:
    
        mlflow.log_params(params)

    # define 3-fold cross validation test harness
    kfold = KFold(n_splits=3, shuffle=True, random_state=11)

    cvscores = []
    for train, test in kfold.split(X_train, y_train):
      # create model
        model = build(lr=lr,
                      init= init,
                      activ= params.get("activation"),
                      optim=params.get("optimizer"),
                      input_shape= params.get("input_shape"))

        # Fit the model
        model.fit(X_train[train], y_train[train], epochs=params.get("epochs"), batch_size=params.get("batch_size"), verbose=0)
        # evaluate the model
        scores = model.evaluate(X_train[test], y_train[test], verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        K.clear_session()

    # upload model
    mlflow.keras.log_model(model, artifact_path=run_name)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
