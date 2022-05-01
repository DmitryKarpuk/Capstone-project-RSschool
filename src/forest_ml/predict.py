import mlflow
import click
import pandas as pd
from pathlib import Path
from joblib import load
from . import __version__
from forest_ml.data import get_data


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

    if Path(model).suffix == ".joblib":
        # Load model from joblib file.
        loaded_model = load(model)
    else:
        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(model)
    # Load data
    data_df = get_data(dataset_path)
    # Predict on a Pandas DataFrame.
    pred = loaded_model.predict(pd.DataFrame(data_df.drop(columns="Id")))
    submission = pd.DataFrame(data={"Id": data_df["Id"], "Cover_Type": pred})
    submission.to_csv(submission_path, index=False)
