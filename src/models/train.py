import joblib
from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# =====================================================
# Config
# =====================================================
TARGET_COL = "Churn"
ARTIFACTS_DIR = Path("artifacts/models")
METRICS_DIR = Path("artifacts/metrics")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Training Function
# =====================================================
def train_models(df: pd.DataFrame):

    # ---------------------------
    # Safety checks (industry)
    # ---------------------------
    if TARGET_COL not in df.columns:
        raise RuntimeError("Training aborted: target column missing")

    y = df[TARGET_COL]

    if y.isna().any():
        raise RuntimeError("Training aborted: target contains NaN")

    # Normalize labels if needed
    if set(y.unique()) <= {"Yes", "No"}:
        y = y.map({"Yes": 1, "No": 0})

    if not set(y.unique()).issubset({0, 1}):
        raise RuntimeError("Training aborted: target must be binary 0/1")

    X = df.drop(columns=[TARGET_COL])

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    # ---------------------------
    # Preprocessing
    # ---------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols),
        ]
    )

    # ---------------------------
    # Train-test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------------------
    # Candidate models
    # ---------------------------
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", LogisticRegression(max_iter=2000))
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced"
                ))
            ]
        ),
    }

    results = {}

    # ---------------------------
    # Train & Evaluate
    # ---------------------------
    for name, model in models.items():
        print(f"Training model: {name}")

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds)),
            "recall": float(recall_score(y_test, preds)),
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)),
        }

        print(f"Metrics for {name}: {metrics}")

        # Save individual model
        joblib.dump(model, ARTIFACTS_DIR / f"{name}.joblib")

        results[name] = {
            "model": model,
            "metrics": metrics
        }

    # ---------------------------
    # Select Best Model (by ROC-AUC)
    # ---------------------------
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]

    print("\nBest model selected:", best_name)
    print("Best metrics:", best_metrics)

    # ---------------------------
    # Save BEST pipeline for deployment
    # ---------------------------
    joblib.dump(best_model, ARTIFACTS_DIR / "churn_pipeline.joblib")

    # Save metrics to JSON
    with open(METRICS_DIR / "evaluation.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "metrics": best_metrics
            },
            f,
            indent=4
        )

    # Collect only metrics for table printing
    all_metrics = {name: results[name]["metrics"] for name in results}

    return best_model, X_test, y_test, best_metrics, all_metrics
