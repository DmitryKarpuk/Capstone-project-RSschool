import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List


def get_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def get_split_data(
    df: pd.DataFrame, features: List, target: List
) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df.loc[:, features]
    y = df.loc[:, target].values.ravel()
    return X, y


def get_partial_data(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
  rng = np.random.default_rng()
  new_len = int(len(df) * ratio)
  rows = rng.choice(len(df), new_len, replace=False)
  return df.iloc[rows, :]