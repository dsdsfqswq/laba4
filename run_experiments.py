import json
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from ensemble_models import (
    _add_domain_features,
    _build_preprocessor,
    run_ensemble_experiment,
)
from grader_contracts.ensembles import EnsembleInput


def main():
    df = pd.read_csv("german.csv", sep=";")
    X = df.drop(columns=["Creditability"])
    y = df["Creditability"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_fe = _add_domain_features(X_train)
    X_test_fe = _add_domain_features(X_test)
    prep = _build_preprocessor(X_train_fe)

    candidates = {
        "RandomForest_baseline": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "RandomForest_tuned": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
        ),
        "ExtraTrees_tuned": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            random_state=42,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        ),
        "MLP_64_32": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1.0,
            learning_rate_init=1e-3,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=42,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    experiments = []

    for name, clf in candidates.items():
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        cv_auc = cross_val_score(pipe, X_train_fe, y_train, cv=cv, scoring="roc_auc").mean()

        pipe.fit(X_train_fe, y_train)
        proba = pipe.predict_proba(X_test_fe)[:, 1]
        pred = (proba >= 0.5).astype(int)

        experiments.append({
            "model_name": name,
            "cv_roc_auc": round(float(cv_auc), 4),
            "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        })

    contract_input = EnsembleInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
        random_seed=42,
    )
    final_res = run_ensemble_experiment(contract_input)

    final_voting = VotingClassifier(
        estimators=[
            ("hgb", candidates["HistGradientBoosting"]),
            ("rf", candidates["RandomForest_tuned"]),
            ("mlp", candidates["MLP_64_32"]),
        ],
        voting="soft",
        weights=(2, 2, 1),

    )
    final_cv_auc = cross_val_score(
        Pipeline([("prep", prep), ("clf", final_voting)]),
        X_train_fe,
        y_train,
        cv=cv,
        scoring="roc_auc",
    ).mean()

    experiments.append({
        "model_name": final_res.model_name,
        "cv_roc_auc": round(float(final_cv_auc), 4),
        "test_roc_auc": round(final_res.roc_auc, 4),
        "accuracy": round(final_res.accuracy, 4),
        "precision": round(final_res.precision, 4),
        "recall": round(final_res.recall, 4),
        "f1": round(final_res.f1, 4),
    })

    output_payload = {
        "dataset": "german.csv",
        "split": {"test_size": 0.2, "stratify": True, "random_seed": 42},
        "selection_criterion": "cv_roc_auc",
        "selected_model": final_res.model_name,
        "final_test_metrics": {
            "roc_auc": round(final_res.roc_auc, 4),
            "accuracy": round(final_res.accuracy, 4),
            "precision": round(final_res.precision, 4),
            "recall": round(final_res.recall, 4),
            "f1": round(final_res.f1, 4),
        },
        "experiments": experiments,
    }

    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(pd.DataFrame(experiments).to_markdown(index=False))


if __name__ == "__main__":
    main()
