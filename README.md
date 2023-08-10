# Airflow DAGs for Bookdown Project *MLOps Engineering*

This repository contains a collection of Airflow DAGs for solving various ML problems. These DAGs are designed to provide a scalable and reliable pipeline for machine learning tasks, leveraging the power of Apache Airflow and integrating with MLflow.

## Use Cases

### 1. CNN Skin Cancer Classification (cnn_skin_cancer)

The `cnn_skin_cancer` use case focuses on the classification of melanoma as benign or malignant using a TensorFlow model. It sets up an containerized Airflow DAG pipeline with MLflow integration, allowing for efficient and reproducible machine learning workflows.

## Usage

To use the DAGs in this repository, follow the steps below:

1. Install Airflow, Docker, and MLflow
2. Clone this repository: `git clone https://github.com/seblum/mlops-airflow-dags.git`
3. Navigate to the cloned repository: `cd mlops-airflow-dags`.
4. Set up a virtualEnv and install the `requirements.txt`
5. Set the following environment variables:
```bash
export AWS_ACCESS_KEY_ID="<AWS-ACCESS-KEY>"
export AWS_SECRET_ACCESS_KEY="<AWS-SECRET-ACCESS-KEY>"
export AWS_ROLE_NAME="<AWS-ROLE-WITH-RELEVANT-ACCESS-TO-S3>"
export AWS_BUCKET="<S3-BUCKET-WITH-DATA>"
export AWS_REGION="<AWS-REGION>"
```
7. Customize the DAGs to fit your specific requirements by modifying the DAG definition files.
8. Run MLflow: `mlflow ui -p 5008`.
9. Run the Airflow webserver: `airflow webserver -p 8081`.
10. Run the Airflow scheduler: `airflow scheduler`.
11. Access the Airflow web interface by opening `http://localhost:8081` in your web browser.
12. Configure and trigger the desired DAGs through the Airflow UI.

## Contributing

Contributions to this repository are welcome! If you have any improvements, bug fixes, or new use case suggestions, please submit a pull request. For major changes, please open an issue first to discuss the proposed changes.

##  License

This repository is licensed under the Apache License. Feel free to use and modify the code as per your needs.

This project is related to the Bookdown Book [MLOps Engineering](https://github.com/seblum/mlops-engineering-book) and the ML platform based on Airflow & MLflow of [this project](https://github.com/seblum/mlops-airflow-on-eks).

## Acknowledgements

I would like to express my gratitude to the authors and contributors of multiple online resources that inspired and helped this project. Their valuable insights and guidance are greatly appreciated.

If you find this repository helpful, consider giving it a ⭐️ to show your appreciation!
