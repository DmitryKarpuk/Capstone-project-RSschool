import requests
import pandas as pd
from pathlib import Path
import click

URL = 'http://localhost:9696/predict'

@click.command
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
def test_app(data_path: Path, index: int) -> None:
    forest = pd.read_csv(data_path).iloc[[index], 1:].to_dict()
    response = requests.post(URL, json=forest)
    result = response.json()
    print(result)

# if __name__ == "__main__":
#     predict_test()