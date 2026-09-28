import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_clean, split_features_target

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
    }


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, y_pred, output_dict=True)
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision_attrition": report["1"]["precision"],
        "recall_attrition": report["1"]["recall"],
        "f1_attrition": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def run(save: bool = True) -> dict:
    df = load_clean()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    best_name, best_pipeline, best_score = None, None, -1.0

    for name, model in candidate_models().items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)
        results[name] = metrics

        if metrics["roc_auc"] > best_score:
            best_name, best_pipeline, best_score = name, pipeline, metrics["roc_auc"]

    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(best_pipeline, MODELS_DIR / "attrition_model.joblib")
        (MODELS_DIR / "metrics.json").write_text(
            json.dumps({"best_model": best_name, "results": results}, indent=2), encoding="utf-8"
        )

    return {"best_model": best_name, "results": results}


if __name__ == "__main__":
    summary = run()
    print(f"Meilleur modèle : {summary['best_model']}")
    print(json.dumps(summary["results"], indent=2))
