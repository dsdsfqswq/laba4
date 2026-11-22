from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

CATEGORICAL_COLS = [
    "Account_Balance",
    "Payment_Status_of_Previous_Credit",
    "Purpose",
    "Value_Savings_Stocks",
    "Length_of_current_employment",
    "Sex_Marital_Status",
    "Guarantors",
    "Most_valuable_available_asset",
    "Concurrent_Credits",
    "Type_of_apartment",
    "Occupation",
    "Telephone",
    "Foreign_Worker",
]


def _to_dataframe(features: object) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.copy()
    arr = np.asarray(features)
    cols = [f"f_{i}" for i in range(arr.shape[-1])]
    return pd.DataFrame(arr, columns=cols)


def _add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    if {"Credit_Amount", "Duration_of_Credit_monthly"}.issubset(X.columns):
        amt = pd.to_numeric(X["Credit_Amount"], errors="coerce")
        dur = pd.to_numeric(X["Duration_of_Credit_monthly"], errors="coerce")
        X["monthly_payment_burden"] = amt / (dur + 1e-5)
        X["log_credit_amount"] = np.log1p(np.clip(amt, a_min=0, a_max=None))

    if {"Credit_Amount", "Age_years"}.issubset(X.columns):
        amt = pd.to_numeric(X["Credit_Amount"], errors="coerce")
        age = pd.to_numeric(X["Age_years"], errors="coerce")
        X["credit_to_age_ratio"] = amt / (age + 1e-5)

    return X


def _build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    known_cats = [c for c in CATEGORICAL_COLS if c in X_train.columns]
    obj_cats = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = list(dict.fromkeys(known_cats + obj_cats))
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    transformers = []

    if num_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    seed = int(data.random_seed)

    X_train = _add_domain_features(_to_dataframe(data.train_features))
    X_test = _add_domain_features(_to_dataframe(data.test_features))
    y_train = np.asarray(data.train_target)
    y_test = np.asarray(data.test_target)

    preprocessor = _build_preprocessor(X_train)

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return EnsembleResult(
        model_name="RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5)",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
