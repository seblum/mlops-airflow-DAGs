# import matplotlib.pyplot as plt
# import mlflow
# import numpy as np
# from keras import backend as K
# from keras import layers
# from keras.callbacks import ReduceLROnPlateau
# from keras.models import Sequential
# from keras.optimizers import Adam, RMSprop
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import KFold
# from tensorflow import keras

# # ----- ----- ----- ----- ----- -----
# # CROSS VALIDATION

# K.clear_session()


# def train_crossval_model(mlflow_tracking_uri: str, mlflow_experiment_id: str, **kwargs):

#         # define 3-fold cross validation test harness
#         kfold = KFold(n_splits=3, shuffle=True, random_state=11)

#         cvscores = []
#         for train, test in kfold.split(X_train, y_train):
#             # create model

#             # basically a BasicNet()
#             model =

#             model.compile(optimizer=params.get("optimizer"), loss=params.get("loss"), metrics=params.get("metrics"))
#             # Fit the model
#             model.fit(
#                 X_train[train],
#                 y_train[train],
#                 epochs=params.get("epochs"),
#                 batch_size=params.get("batch_size"),
#                 verbose=0,
#             )

#             # evaluate the model
#             scores = model.evaluate(X_train[test], y_train[test], verbose=0)
#             print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
#             cvscores.append(scores[1] * 100)
#             K.clear_session()

#         model_uri = f"runs:/{run_id}/{run_name}"

#         mlflow.keras.log_model(model, artifact_path=run_name)

#         print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

#         # Testing model on test data to evaluate
#         y_pred = model.predict(X_test)
#         prediction_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
#         mlflow.log_metric("prediction_accuracy", prediction_accuracy)
#         print(prediction_accuracy)

#         mlflow.keras.log_model(model, artifact_path=run_name)

#         mv = mlflow.register_model(model_uri, run_name)
#         print("Name: {}".format(mv.name))
#         print("Version: {}".format(mv.version))
#         print("Stage: {}".format(mv.current_stage))

#     kwargs["ti"].xcom_push(key=f"run_id-{run_name}", value=run_id)
