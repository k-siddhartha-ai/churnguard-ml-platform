import pandas as pd
from pathlib import Path
from src.data.ingest import load_raw_data
from src.data.validation import validate_data
from src.data.cleaning import clean_data
from src.models.train import train_models
from src.evaluation.metrics import evaluate_model  # or evaluate_model (confirm name)

from src.explainability.explainability import (
    logistic_coefficients,
    random_forest_importance,
)


def run_pipeline():
    # =====================================================
    # 1️⃣ Load raw data
    # =====================================================
    df = load_raw_data("data/raw/telco_churn.csv")

    # =====================================================
    # 2️⃣ Validate schema & integrity
    # =====================================================
    validate_data(df)

    # =====================================================
    # 3️⃣ Clean data
    # =====================================================
    df = clean_data(df)

    # =====================================================
    # 4️⃣ Train models
    # =====================================================
    best_model, X_test, y_test, best_metrics, all_metrics = train_models(df)
    # =====================================================
    # 📊 MODEL COMPARISON TABLE (ALL MODELS)
    # =====================================================
    df_metrics = pd.DataFrame(all_metrics).T
    df_metrics = df_metrics[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    df_metrics = df_metrics.round(4)

    print("\nMODEL COMPARISON TABLE\n")
    print(df_metrics.to_string())

    # Save comparison table (resume-quality)
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(reports_dir / "model_comparison_metrics.csv")

    # =====================================================
    # =====================================================
    # 5️⃣ Evaluate BEST model (SAVE REPORTS)
    # =====================================================
    reports_dir = Path("artifacts/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # We evaluate only the BEST model (already selected)
    metrics = evaluate_model("churn_pipeline", X_test, y_test)

    print("\nBEST MODEL RESULTS")
    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1-score:", metrics["f1_score"])

    # Save summary report
    with open(reports_dir / "best_model_summary.txt", "w") as f:
        f.write(f"Accuracy: {metrics['accuracy']}\n")
        f.write(f"Precision: {metrics['precision']}\n")
        f.write(f"Recall: {metrics['recall']}\n")
        f.write(f"F1-score: {metrics['f1_score']}\n")

    # Save confusion matrix
    cm_df = pd.DataFrame(metrics["confusion_matrix"])
    cm_df.to_csv(reports_dir / "best_model_confusion_matrix.csv", index=False)

    # =====================================================
    # 6️⃣ Explainability (BEST MODEL ONLY — INDUSTRY STYLE)
    # =====================================================
    explain_dir = Path("artifacts/explainability")
    explain_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = best_model.named_steps["preprocessing"]
    feature_names = preprocessor.get_feature_names_out()

    model_name = type(best_model.named_steps["model"]).__name__.lower()

    # Logistic Regression explainability
    if "logistic" in model_name:
        explain_df = logistic_coefficients(best_model, feature_names)
        explain_df.to_csv(explain_dir / "logistic_coefficients.csv", index=False)
        print("✅ Logistic Regression explainability saved.")

    # Random Forest explainability
    elif "forest" in model_name:
        explain_df = random_forest_importance(best_model, feature_names)
        explain_df.to_csv(explain_dir / "random_forest_importance.csv", index=False)
        print("✅ Random Forest explainability saved.")

    else:
        print("⚠️ Explainability not supported for this model type.")




    print("✅ Explainability artifacts saved.")
    print("✅ Evaluation reports saved.")
    print("✅ Explainability artifacts saved to artifacts/explainability/")


# =====================================================
# Entry point (VERY IMPORTANT)
# =====================================================
if __name__ == "__main__":
    run_pipeline()

