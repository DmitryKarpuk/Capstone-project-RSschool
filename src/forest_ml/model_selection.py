from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from typing import Dict, List
import numpy as np
import pandas as pd


def nested_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    refit: str,
    pipeline: Pipeline,
    seed: int,
    params: Dict,
    scores: Dict,
) -> Dict:
    best_pipelines: list = []
    best_params: list = []
    cross_vall_outer = KFold(n_splits=cv, shuffle=True, random_state=seed)
    for train, test in cross_vall_outer.split(X):
        X_train, X_test = (X.iloc[train, :], X.iloc[test, :])
        y_train, y_test = (y[train], y[test])
        cross_vall_inner = KFold(n_splits=cv, shuffle=True, random_state=seed)
        gscv = GridSearchCV(
            pipeline,
            params,
            scoring=refit,
            cv=cross_vall_inner,
            refit=True,
        )
        result = gscv.fit(X_train, y_train)
        best_pipelines.append(result.best_estimator_)
        best_params.append(result.best_params_)
        pred = result.best_estimator_.predict(X_test)
        pre_prob = result.best_estimator_.predict_proba(X_test)
        scores["accuracy"] = np.append(
            scores["accuracy"], accuracy_score(y_test, pred)
        )
        scores["f1_weighted"] = np.append(
            scores["f1_weighted"],
            f1_score(y_test, pred, average="weighted"),
        )
        scores["roc_auc_ovo"] = np.append(
            scores["roc_auc_ovo"],
            roc_auc_score(y_test, pre_prob, multi_class="ovo"),
        )
    best_index = np.argmax(scores[refit])
    cv_result = {
        "accuracy": scores["accuracy"][best_index],
        "f1": scores["f1_weighted"][best_index],
        "roc_auc": scores["roc_auc_ovo"][best_index],
        "estimator": best_pipelines[best_index].fit(X, y),
        "best_param": best_params[best_index],
    }
    return cv_result


def grid_search_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    refit: str,
    pipeline: Pipeline,
    seed: int,
    params: Dict,
    scoring: List,
) -> Dict:
    cross_vall = KFold(n_splits=cv, shuffle=True, random_state=seed)
    gscv = GridSearchCV(
        pipeline,
        params,
        scoring=scoring,
        cv=cross_vall,
        refit=refit,
    )
    result = gscv.fit(X, y)
    best_index = result.best_index_
    cv_result = {
        "accuracy": result.cv_results_["mean_test_accuracy"][best_index],
        "f1": result.cv_results_["mean_test_f1_weighted"][best_index],
        "roc_auc": result.cv_results_["mean_test_roc_auc_ovo"][best_index],
        "estimator": result.best_estimator_.fit(X, y),
        "best_param": result.best_params_,
    }
    return cv_result


def kfold_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int,
    pipeline: Pipeline,
    seed: int,
    params: Dict,
    scores: Dict,
) -> Dict:
    cross_vall = KFold(n_splits=cv, shuffle=True, random_state=seed)
    pipeline.set_params(**params)
    for train, test in cross_vall.split(X):
        X_train, X_test = (X.iloc[train, :], X.iloc[test, :])
        y_train, y_test = (y[train], y[test])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        pre_prob = pipeline.predict_proba(X_test)
        scores["accuracy"] = np.append(
            scores["accuracy"], accuracy_score(y_test, pred)
        )
        scores["f1_weighted"] = np.append(
            scores["f1_weighted"],
            f1_score(y_test, pred, average="weighted"),
        )
        scores["roc_auc_ovo"] = np.append(
            scores["roc_auc_ovo"],
            roc_auc_score(y_test, pre_prob, multi_class="ovo"),
        )
        cv_result = {
            "accuracy": np.mean(scores["accuracy"]),
            "f1": np.mean(scores["f1_weighted"]),
            "roc_auc": np.mean(scores["roc_auc_ovo"]),
            "estimator": pipeline.fit(X, y),
            "best_param": params,
        }
    return cv_result
