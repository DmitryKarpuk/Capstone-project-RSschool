from email.policy import default
from pathlib import Path
from turtle import pd
import click
from joblib import dump

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

@click.command()
@click.option('-d', '--dataset-path', default='data/train.csv',
                    type=click.Path(exists=True, dir_okay=False,
                                                path_type=Path))
@click.optin('-s', '--save-model-path', default='data/model.joblib',
                                            type=click.Path(exists=True, dir_okay=False,
                                                path_type=Path))
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    show_default=True
)
@click.option(
    "--min-sample_split",
    default=2,
    type=int,
    show_default=True
)
@click.option(
    "--min-sample-leaf",
    default=2,
    type=int,
    show_default=True
)
# @click.option(
#     "--cross-val-type",
#     default='K-fold',
#     type=str,
#     show_default=True,
#     help='Type of cross validation'
)
@click.option(
    "--cv",
    default=5,
    type=int,
    show_default=True,
    help='Cross-validation splitting strategy'
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_sample_split: int,
    min_sample_leaf: int,    
    cv: int
) -> None:
    df=pd.read_csv(dataset_path, index_col='Id')
    X = df.iloc[:,:-1]
    y = df.iloc[:, -1]
    scorring = ('accuracy', 'f1', 'roc_auc')
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth,
    min_sample_split=min_sample_split, min_sample_leaf=min_sample_leaf)
    cv_results = cross_validate(clf, X, y, cv=cv, scoring=scorring)
