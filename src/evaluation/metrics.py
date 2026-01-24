# src/evaluation/metrics.py

from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

ARTIFACTS_DIR = Path("artifacts/models")
REPORTS_DIR = Path("artifacts/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model_name: str, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Industry-grade evaluation:
    - Loads trained pipeline
    - Computes metrics
    - Saves:
        * metrics CSV
        * classification report TXT
        * confusion matrix IMAGE
        * ROC curve IMAGE
    """

    # ---------------------------
    # Load trained model
    # ---------------------------
    model_path = ARTIFACTS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise RuntimeError(f"Model not found: {model_path}")

    model = joblib.load(model_path)

    # ---------------------------
    # Predictions
    # ---------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ---------------------------
    # Metrics
    # ---------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # ---------------------------
    # Save metrics CSV
    # ---------------------------
    metrics_df = pd.DataFrame(
        {
            "accuracy": [acc],
            "precision": [prec],
            "recall": [rec],
            "f1_score": [f1],
            "roc_auc": [roc_auc],
        }
    )

    metrics_file = REPORTS_DIR / f"{model_name}_metrics.csv"
    report_file = REPORTS_DIR / f"{model_name}_report.txt"

    metrics_df.to_csv(metrics_file, index=False)

    # ---------------------------
    # Save classification report
    # ---------------------------
    with open(report_file, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # ---------------------------
    # 🔥 Save Confusion Matrix IMAGE
    # ---------------------------
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.colorbar()

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    cm_image_path = REPORTS_DIR / f"{model_name}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_image_path)
    plt.close()

    # ---------------------------
    # 🔥 Save ROC Curve IMAGE
    # ---------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    roc_image_path = REPORTS_DIR / f"{model_name}_roc_curve.png"
    plt.tight_layout()
    plt.savefig(roc_image_path)
    plt.close()

    # ---------------------------
    # Final logs
    # ---------------------------
    print(f"Evaluation completed for: {model_name}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Report saved to: {report_file}")
    print(f"Confusion matrix image saved to: {cm_image_path}")
    print(f"ROC curve image saved to: {roc_image_path}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }