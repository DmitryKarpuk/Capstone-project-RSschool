from pathlib import Path
import click
from joblib import dump

import mlflow
import mlflow.sklearn

import numpy as np
import json

from .data import get_partial_data, get_data, get_split_data
from .pipeline import create_pipeline
from .model_selection import kfold_cv, nested_cv, grid_search_cv

from . import __version__
import warnings

warnings.filterwarnings("ignore")

mlflow.set_tracking_uri("http://localhost:5000")


@click.command(
    help="Trains an model on forest cover type dataset."
    "The input is expected in csv format."
    "The model and its metrics are logged with mlflow."
)
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
@click.option(
    "-p",
    "--param-path",
    default="config/tuning_params.json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to file with model parameters.",
)
@click.option(
    "-m",
    "--model",
    default="logisticregression",
    type=click.Choice(["logisticregression", "randomforest"]),
    show_default=True,
)
@click.option(
    "-st",
    "--selection-type",
    default="grid_search_cv",
    type=click.Choice(["nested_cv", "grid_search_cv", "kfold_cv"]),
    show_default=True,
    help="Method of estimating model",
)
@click.option(
    "--refit",
    default="accuracy",
    type=click.Choice(["accuracy", "f1_weighted", "roc_auc_ovo"]),
    show_default=True,
)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--cv",
    default=5,
    type=int,
    show_default=True,
    help="Cross-validation splitting strategy",
)
@click.option(
    "--preprocessor",
    default="none",
    type=click.Choice(["none", "PCA", "SelectFromModel"]),
    show_default=True,
    help="Feature engineering methods",
)
@click.option(
    "--components",
    default=2,
    type=int,
    show_default=True,
    help="Number of components to keep",
)
@click.option(
    "--data-ratio",
    default=1,
    type=click.FloatRange(0, 1, min_open=True),
    show_default=True,
    help="Ratio of data for train",
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    param_path: Path,
    model: str,
    selection_type: str,
    refit: str,
    seed: int,
    cv: int,
    preprocessor: str,
    components: int,
    use_scaler: bool,
    data_ratio: float,
) -> None:
    with open(param_path) as json_file:
        params = json.load(json_file)[model]

    scores: dict = {
        "accuracy": np.array([]),
        "f1_weighted": np.array([]),
        "roc_auc_ovo": np.array([]),
    }
    data_df = get_partial_data(
        get_data(dataset_path).drop(columns="Id"),
        ratio=data_ratio,
        seed=seed,
    )
    features = list(data_df.columns)[:-1]
    target = list(data_df.columns)[-1]
    X, y = get_split_data(data_df, features, target)
    pipeline = create_pipeline(
        method=preprocessor,
        model=model,
        n_components=components,
        use_scaler=use_scaler,
        seed=seed,
    )
    if selection_type == "nested_cv":
        cv_result = nested_cv(X, y, cv, refit, pipeline, seed, params, scores)
    elif selection_type == "grid_search_cv":
        cv_result = grid_search_cv(
            X, y, cv, refit, pipeline, seed, params, list(scores.keys())
        )
    else:
        cv_result = kfold_cv(X, y, cv, pipeline, seed, params, scores)
    accuracy = cv_result["accuracy"]
    f1 = cv_result["f1"]
    roc_auc = cv_result["roc_auc"]
    estimator = cv_result["estimator"]
    best_param = cv_result["best_param"]
    with mlflow.start_run():
        if model == "logisticregression":
            mlflow.log_param(
                "max_iter", best_param["logisticregression__max_iter"]
            )
            mlflow.log_param(
                "penalty", best_param["logisticregression__penalty"]
            )
            mlflow.log_param("logreg_c", best_param["logisticregression__C"])
        else:
            mlflow.log_param(
                "n_estimators", best_param["randomforest__n_estimators"]
            )
            mlflow.log_param("max_depth", best_param["randomforest__max_depth"])
            mlflow.log_param(
                "min_samples_split",
                best_param["randomforest__min_samples_split"],
            )
            mlflow.log_param(
                "min_samples_leaf", best_param["randomforest__min_samples_leaf"]
            )
        mlflow.log_param("random_state", seed)
        if preprocessor != "none":
            mlflow.log_param("preprocessor", preprocessor)
            if preprocessor == "PCA":
                mlflow.log_param("components", components)
        if use_scaler:
            mlflow.log_param("use_scaler", "StandardScaler")
        mlflow.log_param("data ratio", data_ratio)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(estimator, model)
    click.echo(click.style("<<Metrics>>", fg="green"))
    click.echo(f"Accuracy: {accuracy}")
    click.echo(f"F1: {f1}")
    click.echo(f"ROC_AUC: {roc_auc}")
    click.echo(click.style("<<Model>>", fg="green"))
    click.echo(f"Best model: {estimator}")
    dump(estimator, save_model_path)
    click.echo(f"Model is saved to {save_model_path}")
