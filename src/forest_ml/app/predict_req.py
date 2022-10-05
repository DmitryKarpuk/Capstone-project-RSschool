import requests
import pandas as pd
from pathlib import Path
import click

URL = "http://localhost:9696/predict"


@click.command(help="Script with request for testing model app.")
@click.option(
    "-p",
    "--data-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-i",
    "--index",
    default=0,
    type=int,
)
def predict_req(data_path: Path, index: int) -> None:
    forest = pd.read_csv(data_path).iloc[[index], 1:].to_dict()
    response = requests.post(URL, json=forest)
    result = response.json()
    print(result)
