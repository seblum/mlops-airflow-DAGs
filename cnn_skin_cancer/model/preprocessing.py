import os
import numpy as np
import keras
from keras.utils.np_utils import to_categorical  # used for converting labels to one-hot-encoding
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from PIL import Image
import pathlib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import mlflow


def run_preprocessing(mlflow_tracking_uri: str, mlflow_experiment_id: str, **kwargs) -> None:

    dir_path = pathlib.Path(__file__).parent.absolute()
    print(dir_path)
    parent_path = dir_path.parent
    print(parent_path)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    # TODO timestamp in name
    run_name = "preprocessing"
    with mlflow.start_run(experiment_id=mlflow_experiment_id, run_name=run_name) as run:

        DATAPATH = f"{parent_path}/data/"

        folder_benign_train = f"{DATAPATH}train/benign"
        folder_malignant_train = f"{DATAPATH}train/malignant"

        folder_benign_test = f"{DATAPATH}test/benign"
        folder_malignant_test = f"{DATAPATH}test/malignant"

        read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

        def _load_and_convert_images(folder_path: str):
            ims = [read(os.path.join(folder_path, filename)) for filename in os.listdir(folder_path)]
            return np.array(ims, dtype="uint8")

        def _create_label(x_dataset: np.array):
            return np.zeros(x_dataset.shape[0])

        def _merge_data(set_one: np.array, set_two: np.array):
            return np.concatenate((set_one, set_two), axis=0)

        # TODO: How to do images in airflow?
        # def _display_image(X_train, y_train):
        #     # Display first 15 images of moles, and how they are classified
        #     # This image can be logged and stored in mlflow
        #     w = 40
        #     h = 30
        #     fig = plt.figure(figsize=(12, 8))
        #     columns = 5
        #     rows = 3

        #     for i in range(1, columns * rows + 1):
        #         ax = fig.add_subplot(rows, columns, i)
        #         if (y_train[i] == 0).any():
        #             ax.title.set_text("Benign")
        #         else:
        #             ax.title.set_text("Malignant")
        #         plt.imshow(X_train[i], interpolation="nearest")
        #     mlflow.log_figure(fig, "exemplary_images.png")

        # Load in training pictures
        X_benign = _load_and_convert_images(folder_benign_train)
        X_malignant = _load_and_convert_images(folder_malignant_train)
        X_train = _merge_data(X_benign, X_malignant)

        # Load in testing pictures
        X_benign_test = _load_and_convert_images(folder_benign_test)
        X_malignant_test = _load_and_convert_images(folder_malignant_test)
        X_test = _merge_data(X_benign_test, X_malignant_test)

        # Create labels
        y_benign = _create_label(X_benign)
        y_malignant = _create_label(X_malignant)
        y_train = _merge_data(y_benign, y_malignant)

        y_benign_test = _create_label(X_benign_test)
        y_malignant_test = _create_label(X_malignant_test)
        y_test = _merge_data(y_benign_test, y_malignant_test)

        # Shuffle data
        X_train, y_train = shuffle(X_train, y_train)
        X_test, y_test = shuffle(X_test, y_test)

        y_train = to_categorical(y_train, num_classes=2)
        y_test = to_categorical(y_test, num_classes=2)

        # _display_image(X_train, y_train)

        # With data augmentation to prevent overfitting
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        np.save(f"{parent_path}/X_train.npy", X_train)
        np.save(f"{parent_path}/y_train.npy", y_train)
        np.save(f"{parent_path}/X_test.npy", X_test)
        np.save(f"{parent_path}/y_test.npy", y_test)

        kwargs["ti"].xcom_push(key="path_X_train", value=f"{parent_path}/X_train.npy")
        kwargs["ti"].xcom_push(key="path_y_train", value=f"{parent_path}/y_train.npy")
        kwargs["ti"].xcom_push(key="path_X_test", value=f"{parent_path}/X_test.npy")
        kwargs["ti"].xcom_push(key="path_y_test", value=f"{parent_path}/y_test.npy")
