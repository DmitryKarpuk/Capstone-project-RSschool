from dis import show_code
from pathlib import Path
import click
from joblib import dump

import mlflow
import mlflow.sklearn

import numpy as np


from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .data import get_partial_data, get_data, get_split_data
from .pipeline import create_pipeline

from . import __version__

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
    "--model",
    default="LogisticRegression",
    type=click.Choice(["LogisticRegression", "RandomForestClassifier"]),
    show_default=True,
)
@click.option(
    "--metric",
    default="accuracy",
    type=click.Choice(["accuracy", "f1_weighted", "roc_auc_ovo"]),
    show_default=True,
)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option(
    "--use-scaler", default=True, type=bool, show_default=True,
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
    type=click.FloatRange(0, 1, max_open=True),
    show_default=True,
    help="Ratio of data for train",
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    model: str,
    metric: str,
    seed: int,
    cv: int,
    preprocessor: str,
    components: int,
    use_scaler: bool,
    data_ratio: float,
) -> None:
    if model == "LogisticRegression":
        params = {
            "logisticregression__penalty": ["l1", "l2"],
            "logisticregression__max_iter": [1000, 2000],
            "logisticregression__C": [1, 1.5, 2],
            "logisticregression__random_state": [seed],
            "logisticregression__solver": ["saga"],
        }
    else:
        params = {
            "randomforest__n_estimators": [50, 70, 100, 120],
            "randomforest__max_depth": [None, 10, 20, 30],
            "randomforest__min_samples_split": [2, 5, 10],
            "randomforest__min_samples_leaf": [2, 5, 10],
            "randomforest__random_state": [seed],
        }

    with mlflow.start_run():
        cross_vall_outer = KFold(n_splits=cv, shuffle=True, random_state=seed)
        acc_scores: np.ndarray = np.array([])
        f1_scores: np.ndarray = np.array([])
        roc_auc_scores: np.ndarray = np.array([])
        best_params: np.ndarray = np.array([])
        best_pipelines: list = []
        data_df = get_partial_data(
            get_data(dataset_path).drop(columns="Id"),
            ratio=data_ratio,
            seed=seed,
        )
        features = list(data_df.columns)[:-1]
        target = list(data_df.columns)[-1]
        X, y = get_split_data(data_df, features, target)
        for train, test in cross_vall_outer.split(data_df):
            X_train, X_test = (X.iloc[train, :], X.iloc[test, :])
            y_train, y_test = (y[train], y[test])
            cross_vall_inner = KFold(
                n_splits=cv, shuffle=True, random_state=seed
            )
            pipeline = create_pipeline(
                method=preprocessor,
                model=model,
                n_components=components,
                use_scaler=use_scaler,
                seed=seed,
            )
            gscv = GridSearchCV(
                pipeline,
                params,
                scoring=metric,
                cv=cross_vall_inner,
                refit=True,
            )
            result = gscv.fit(X_train, y_train)
            best_pipelines.append(result.best_estimator_)
            best_params = np.append(best_params, result.best_params_)
            pred = result.best_estimator_.predict(X_test)
            acc_scores = np.append(acc_scores, accuracy_score(y_test, pred))
            f1_scores = np.append(
                f1_scores, f1_score(y_test, pred, average="weighted")
            )
            roc_auc_scores = np.append(
                roc_auc_scores,
                roc_auc_score(
                    y_test,
                    result.best_estimator_.predict_proba(X_test),
                    multi_class="ovo",
                ),
            )
        accuracy = np.max(acc_scores)
        f1 = f1_scores[np.argmax(acc_scores)]
        roc_auc = roc_auc_scores[np.argmax(acc_scores)]
        estimator = best_pipelines[np.argmax(acc_scores)].fit(X, y)
        best_param = best_params[np.argmax(acc_scores)]
        click.echo(click.style(f"{best_param}", fg="green"))
        if model == "LogisticRegression":
            mlflow.log_param("max_iter", best_param["logisticregression__max_iter"])
            mlflow.log_param("penalty", best_param["logisticregression__penalty"])
            mlflow.log_param("logreg_c", best_param["logisticregression__C"])
        else:
            mlflow.log_param("n_estimators", best_param["randomforest__n_estimators"])
            mlflow.log_param("max_depth", best_param["randomforest__max_depth"])
            mlflow.log_param(
                "min_samples_split", best_param["randomforest__min_samples_split"]
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
