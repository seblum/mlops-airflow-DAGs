# CNN Skin Cancer Classification with Airflow & MLflow Integration

This repository showcases a use case for CNN skin cancer classification using a TensorFlow model. The project sets up an Airflow DAG pipeline with MLflow integration, providing a scalable and reproducible solution for classifying melanoma as benign or malignant.

# Contents

* `airflow_docker_workflow.py`: This file contains the Airflow DAG definition, which orchestrates the entire pipeline.
* `Docker/`: This directory contains the Dockerfiles used to build the containers for different tasks within the pipeline.
* `data/`: This folder stores the [dataset](https://www.kaggle.com/code/fanconic/cnn-for-skin-cancer-detection
) used for training and evaluation. Although the current script pulls the data from AWS S3, you can place your dataset here for local usage.
* `src/`: This directory contains the ML pipeline code, including data preprocessing, model training, evaluation, and model comparison. The code in this directory is packaged using Poetry, a dependency management tool for Python.

Use Case: CNN Skin Cancer Classification

The `cnn_skin_cancer` use case focuses on training a Convolutional Neural Network (CNN) model to classify melanoma as benign or malignant. The use case leverages the Airflow DAG and MLflow integration to automate the pipeline and track experiment results and metrics.

### Pipeline Steps:

1. Data Preprocessing: The pipeline performs data preprocessing tasks, including data cleaning, augmentation, and normalization, to prepare the skin cancer dataset for training.
2. Model Training & Evaluation: The CNN model is trained on the preprocessed data using the code in the `src/` directory. The trained model is evaluated using various performance metrics to assess its effectiveness in classifying melanoma.
3. Model Comparison: The pipeline compares different models based on their evaluation results to determine the best-performing model.

## Contributing

As for the whole repository, contributions to this repository are welcome! If you have any improvements, bug fixes, or suggestions, feel free to submit a pull request. For major changes, please open an issue first to discuss the proposed changes.
