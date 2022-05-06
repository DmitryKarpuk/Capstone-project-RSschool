from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_transformer
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    method: str, model: str, n_components: int, use_scaler: bool, seed: int
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
                            penalty="l1",
                            solver="saga",
                            max_iter=2000,
                            random_state=seed,
                        )
                    ),
                ),
            )
    if model == "LogisticRegression":
        steps.append(
            ("logisticregression", LogisticRegression(random_state=seed))
        )
    if model == "RandomForestClassifier":
        steps.append(
            ("randomforest", RandomForestClassifier(random_state=seed))
        )
    return Pipeline(steps=steps)
