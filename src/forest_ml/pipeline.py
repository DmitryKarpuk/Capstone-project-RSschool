from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

from typing import Dict

from sklearn.preprocessing import StandardScaler


def create_pipeline(
    method: str, params: Dict, model: str, n_components: int, use_scaler: bool
) -> Pipeline:
    num_features = list(range(10))  # indexes of numeric features
    steps = []
    if use_scaler:
        steps.append(
            (
                "scaler",
                make_column_transformer(
                    (StandardScaler(), num_features),
                    remainder="passthrough",
                ),
            )
        )
    if method != "none":
        if method == "PCA":
            steps.append(
                (
                    "pca",
                    make_column_transformer(
                        (PCA(n_components=n_components), num_features),
                        remainder="passthrough",
                    ),
                )
            )
        if method == "SelectFromModel":
            steps.append(
                (
                    "selector",
                    SelectFromModel(
                        LogisticRegression(
                            penalty="l1", solver="saga", max_iter=2000
                        )
                    ),
                ),
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
