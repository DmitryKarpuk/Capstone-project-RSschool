from pathlib import Path
import click
from joblib import dump

import mlflow
import mlflow.sklearn

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from forest_ml.data import get_data, get_split_data

from . import __version__

mlflow.set_tracking_uri("http://localhost:5000")


@click.command()
@click.version_option(version=__version__)
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
) -> None:
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        cross_vall = KFold(n_splits=cv, shuffle=True, random_state=random_state)
        acc_scores: np.ndarray = np.array([])
        f1_scores: np.ndarray = np.array([])
        roc_auc_scores: np.ndarray = np.array([])
        data_df = get_data(dataset_path).drop(columns="Id")
        features = list(data_df.columns)[:-1]
        target = list(data_df.columns)[-1]
        X, y = get_split_data(data_df, features, target)
        for train, test in cross_vall.split(data_df):
            X_train, X_test = (
                data_df.loc[train, features],
                data_df.loc[test, features],
            )
            y_train, y_test = (
                data_df.loc[train, target].values.ravel(),
                data_df.loc[test, target].values.ravel(),
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc_scores = np.append(acc_scores, accuracy_score(y_test, pred))
            f1_scores = np.append(
                f1_scores, f1_score(y_test, pred, average="weighted")
            )
            roc_auc_scores = np.append(
                roc_auc_scores,
                roc_auc_score(
                    y_test, model.predict_proba(X_test), multi_class="ovo"
                ),
            )
        accuracy = np.mean(acc_scores)
        f1 = np.mean(f1_scores)
        roc_auc = np.mean(roc_auc_scores)
        estimator = model.fit(X, y)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(estimator, "model")
    click.echo(click.style("<<Metrics>>", fg="green"))
    click.echo(f"Accuracy: {accuracy}")
    click.echo(f"F1: {f1}")
    click.echo(f"ROC_AUC: {roc_auc}")
    click.echo(click.style("<<Model>>", fg="green"))
    click.echo(f"Best model: {estimator}")
    dump(estimator, save_model_path)
    click.echo(f"Model is saved to {save_model_path}")
