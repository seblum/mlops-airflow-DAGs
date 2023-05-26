SHELL = /bin/bash

.phony env:
	python3 -m venv .venv-airflow-dags
	source .venv-airflow-dags/bin/activate
	pip3 install -f requirements.txt

run:
	#source .venv-airflow-dags/bin/activate
	mlflow ui -p 5008 &
	echo "> mlflow running"
	airflow standalone & 
	echo "> airflow running"
