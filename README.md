Homework 9 capstone project for RS School Machine Learning course.

[![Badge Status](https://github.com/DmitryKarpuk/9_evaluation_selection/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/DmitryKarpuk/9_evaluation_selection/actions/workflows/tests.yml)

This homework uses [Forest Covered Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

## Usage

This package allows you to train model for predicting forest cover type, predict forest cover type by using fitted model, build EDA using pandas-profiling.

### Preparation
1. Clone this repository to your machine.
2. Download [Forest Covered Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.2.1).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```

### Train
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model> -p <path to json with tuning or estimate params> -m <model for train> -st <method of model selection> 
```
You can configure additional options (such as feature enginireeng or data ratio) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```

### Predict
5. Run predict with the following command:
 ```sh
poetry run predict -d <path to csv with data> -s <path to save result of prediction> -m <path to .joblib or mlflow model> 
```

### Build EDA
5. Run building eda with command:
 ```sh
poetry run eda -d <path to csv with data> -s <path to save html with eda report> 
```

## Development

The code in this repository must be tested, formatted with black, linted with flake8 and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Activate poetry environment:
```
poetry shell
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
poetry run nox [-r]
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
poetry run nox -[r]s black
poetry run black src tests noxfile.py
```
Lint your code with [flake8](https://pypi.org/project/flake8/) by using either nox or poetry:
```
poetry run nox -[r]s lint
poetry run flake8 src tests noxfile.py
```
Check typing of your code with [mypy](https://github.com/python/mypy) by using either nox or poetry:
```
poetry run nox -[r]s lint nox -[r]s mypy
poetry run mypy src tests noxfile.py
```
Test your code with pytest [pytest](https://docs.pytest.org/en/7.1.x/) by using either nox or poetry:
```
nox -[r]s tests
poetry run pytest
```

All necessary screenshots you can find via [link](https://github.com/DmitryKarpuk/9_evaluation_selection/blob/master/reports/report.md)

## Model service.

Model has been deploymented  by flask [link](https://flask.palletsprojects.com/en/2.2.x/). One way to create a WSGI server is to use waitress. This project was packed in a Docker container, you're able to run project on any machine.
In order to run service install docker[link](https://www.docker.com/)
Then build image
```
docker build -t forest_ml_service -f Docker/Dockerfile .
```
Now lets run container
```
docker run -d -p 9696:9696 forest_ml_service:latest
```
As a result, you can test service by using script src/forest_ml/app/predict_req.py.
```
poetry run predict_req [-p-i]
```