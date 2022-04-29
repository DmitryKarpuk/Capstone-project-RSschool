from pathlib import Path
import click
from joblib import dump

import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

mlflow.set_tracking_uri('http://localhost:5000')

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
@click.option("--random-state", default=42, type=int, show_default=True)
@click.option("--n-estimators", default=100, type=int, show_default=True)
@click.option("--max-depth", default=None, type=int, show_default=True)
@click.option("--min-samples_split", default=2, type=int, show_default=True)
@click.option("--min-samples-leaf", default=2, type=int, show_default=True)
@click.option(
    "--key-metric",
    default="accuracy",
    type=str,
    show_default=True,
    help="Metric for select estimator",
)
# @click.option(
#     "--cross-val-type",
#     default='K-fold',
#     type=str,
#     show_default=True,
#     help='Type of cross validation'
# )
@click.option(
    "--cv",
    default=5,
    type=int,
    show_default=True,
    help="Cross-validation splitting strategy",
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    cv: int,
    key_metric: str,
) -> None:
    df = pd.read_csv(dataset_path, index_col="Id")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scorring = ("accuracy", "f1_weighted", "roc_auc_ovr")
    with mlflow.start_run():
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        cv_results = cross_validate(
            clf, X, y, cv=cv, scoring=scorring, return_estimator=True
        )
        accuracy = cv_results["test_accuracy"].max()
        f1 = cv_results["test_f1_weighted"].max()
        roc_auc = cv_results["test_roc_auc_ovr"].max()
        if key_metric == "accuracy":
            estimator = cv_results["estimator"][
                np.argmax(cv_results["test_accuracy"])
            ]
        elif key_metric == "f1":
            estimator = cv_results["estimator"][
                np.argmax(cv_results["test_f1_weighted"])
            ]
        elif key_metric == "roc_auc":
            estimator = cv_results["estimator"][
                np.argmax(cv_results["test_roc_auc_ovr"])
            ]
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('min_samples_split', min_samples_split)
        mlflow.log_param('min_samples_leaf', min_samples_leaf)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1', f1)
        mlflow.log_metric('roc_auc', roc_auc)
        mlflow.sklearn.log_model(estimator, 'model')
    click.echo(click.style("<<Metrics>>", fg="green"))
    click.echo(f"Accuracy: {accuracy}")
    click.echo(f"F1: {f1}")
    click.echo(f"ROC_AUC: {roc_auc}")
    click.echo(click.style("<<Model>>", fg="green"))
    click.echo(f"Best model: {estimator}")
    dump(estimator, save_model_path)
    click.echo(f"Model is saved to {save_model_path}")
