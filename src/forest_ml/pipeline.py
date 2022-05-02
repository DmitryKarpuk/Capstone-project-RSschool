from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA

from typing import Dict

from sklearn.preprocessing import StandardScaler


def create_pipeline(method: str, params: Dict, model: str, n_components: int) -> Pipeline:
    num_features = list(range(11))  # indexes of numeric features
    steps = []
    if method != 'none':
        if method == "StandardScaler":
            steps.append(
                (
                    "scaler",
                    make_column_transformer(
                        (StandardScaler(), num_features),
                        remainder="passthrough",
                    ),
                )
            )
        if method == "PCA":
            steps.append(
                (
                    "scaler",
                    make_column_transformer(
                        (StandardScaler(), num_features),
                        remainder="passthrough",
                    ),
                )
            )
            steps.append(
                (
                    "pca",
                    make_column_transformer(
                        (PCA(n_components=n_components), num_features),
                        remainder="passthrough",
                    ),
                )
            )
    if model == "LogisticRegression":
        steps.append(
            ("logisticregression", LogisticRegression().set_params(**params))
        )
    if model == "RandomForestClassifier":
        steps.append(
            ("randomforest", RandomForestClassifier().set_params(**params))
        )
    return Pipeline(steps=steps)
