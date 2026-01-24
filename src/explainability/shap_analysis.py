import pandas as pd
import numpy as np
import shap
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# =====================================================
# Config
# =====================================================
DATA_PATH = Path("data/raw/telco_churn.csv")
MODEL_PATH = Path("artifacts/models/random_forest.joblib")
OUT_DIR = Path("artifacts/explainability/shap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Churn"
SAMPLE_SIZE = 80  # keeps SHAP fast & stable (industry best practice)

# =====================================================
# Main SHAP Function
# =====================================================
def run_shap_analysis():
    print("🔍 Loading model...")
    model = joblib.load(MODEL_PATH)

    print("📊 Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Drop target if present
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    # -------------------------------------------------
    # Align schema using training pipeline
    # -------------------------------------------------
    preprocessor = model.named_steps["preprocessing"]

    categorical_cols = preprocessor.transformers_[0][2]
    numerical_cols = preprocessor.transformers_[1][2]

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Unknown"

    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0

    df[categorical_cols] = df[categorical_cols].astype(str)
    df[numerical_cols] = (
        df[numerical_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )

    df = df[categorical_cols + numerical_cols]

    # -------------------------------------------------
    # Sample rows (keeps SHAP fast + stable)
    # -------------------------------------------------
    if len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)

    print(f"⚙️ Running SHAP on {len(df)} samples...")

    # -------------------------------------------------
    # Transform using SAME preprocessing as training
    # -------------------------------------------------
    X_processed = preprocessor.transform(df)

    # 🔥 CRITICAL: convert sparse → dense
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    X_processed = X_processed.astype(np.float32)

    # -------------------------------------------------
    # SHAP Tree Explainer (SAFE CONFIG)
    # -------------------------------------------------
    explainer = shap.TreeExplainer(
        model.named_steps["model"],
        feature_perturbation="tree_path_dependent"
    )

    shap_values = explainer.shap_values(
        X_processed,
        check_additivity=False
    )

    # -------------------------------------------------
    # GLOBAL SHAP SUMMARY (FIXED & STABLE)
    # -------------------------------------------------
    print("📈 Saving SHAP summary plot...")

    # 🔥 CRITICAL FIX: handle SHAP output shape correctly
    if isinstance(shap_values, list):
        shap_plot_values = shap_values[1]  # churn = 1
    else:
        shap_plot_values = shap_values

    shap.summary_plot(
        shap_plot_values,
        X_processed,
        plot_type="bar",   # industry-preferred, fast & stable
        show=False
    )

    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary.png", dpi=200)
    plt.close()

    # -------------------------------------------------
    # -----------------------------
    # Handle SHAP output safely
    # -----------------------------
    if isinstance(shap_values, list):
        # Binary classification → positive class
        shap_matrix = shap_values[1]

    elif shap_values.ndim == 3:
        # (samples, features, classes)
        shap_matrix = shap_values[:, :, 1]

    else:
        # Already 2D
        shap_matrix = shap_values

    # Ensure 2D numpy array
    shap_matrix = np.asarray(shap_matrix)

    # -----------------------------
    # Mean absolute SHAP importance
    # -----------------------------
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)

    shap_df = pd.DataFrame({
        "feature_index": np.arange(mean_abs_shap.shape[0]),
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)

    shap_df.to_csv(
        OUT_DIR / "shap_feature_importance.csv",
        index=False
    )

    print("✅ SHAP analysis completed successfully")
    print("📂 Saved to artifacts/explainability/shap/")

# =====================================================
if __name__ == "__main__":
    run_shap_analysis()
