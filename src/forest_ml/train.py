from email.policy import default
from pathlib import Path
from turtle import pd
import click
import pandas as pd

@click.command()
@click.option("-d", "--dataset-path", default="data/train.csv",
                    type=click.Path(exists=True, dir_okay=False,
                                                path_type=Path))
def train(
    dataset_path: Path
) -> None:
    df=pd.read_csv(dataset_path, index_col='Id')
    click.echo(f'Dataset header: {df.columns}')