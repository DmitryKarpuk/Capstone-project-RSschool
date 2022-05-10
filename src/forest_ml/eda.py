import pandas_profiling
import click
from .data import get_data
from pathlib import Path
from . import __version__
import warnings

warnings.filterwarnings("ignore")


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
    "--save-eda-path",
    default="data/eda.html",
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
def eda(dataset_path: Path, save_eda_path: Path) -> None:
    data_df = get_data(dataset_path).drop(columns="Id")
    pandas_profiling.ProfileReport(data_df, minimal=False).to_file(
        save_eda_path
    )
