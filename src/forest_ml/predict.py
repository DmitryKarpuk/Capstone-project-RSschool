import mlflow
import click
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import load
from . import __version__
from forest_ml.data import get_data

def _predict(X: pd.DataFrame, model: str) -> np.array:
    if Path(model).suffix == ".joblib":
        # Load model from joblib file.
        loaded_model = load(model)
    else:
        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(model)
    pred = loaded_model.predict(X)
    return pred

@click.command()
@click.version_option(version=__version__)
@click.option(
    "-d",
    "--dataset-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-s",
    "--submission-path",
    default="data/submission.csv",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-m",
    "--model",
    default="data/model.joblib",
    help="Model path like .joblib or mlflow model path ",
    type=str,
)
def predict(dataset_path: Path, submission_path: Path, model: str) -> None:
    # Load data
    data_df = get_data(dataset_path)
    X = data_df.drop(columns="Id")
    # Predict on a Pandas DataFrame.
    pred = _predict(X, model)
    submission = pd.DataFrame(data={"Id": data_df["Id"], "Cover_Type": pred})
    submission.to_csv(submission_path, index=False)
    click.echo(click.style("<<Model>>", fg="green"))
    click.echo(click.style("<<Prediction>>", fg="green"))
    click.echo(f"Submission is saved to {submission_path}")
    click.echo(f"Submission is saved to {__name__}")
