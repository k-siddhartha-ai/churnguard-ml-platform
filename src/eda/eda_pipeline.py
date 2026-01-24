import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# =====================================================
# Config
# =====================================================
EDA_DIR = Path("artifacts/eda")
EDA_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Churn"

sns.set_theme(style="whitegrid", palette="deep")


# =====================================================
# Main EDA Function
# =====================================================
def run_eda(df: pd.DataFrame) -> None:
    """
    Generates and saves EDA plots for churn analysis.
    Saves plots into artifacts/eda/.
    """

    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError("EDA aborted: 'Churn' column missing")

    # ---------------------------------
    # Normalize target safely
    # ---------------------------------
    df[TARGET_COL] = (
        df[TARGET_COL]
        .astype(str)
        .str.strip()
        .map({"1": "Yes", "0": "No", "Yes": "Yes", "No": "No"})
    )

    df = df.dropna(subset=[TARGET_COL])

    # =====================================================
    # 1️⃣ Churn Distribution
    # =====================================================
    plt.figure(figsize=(6, 4))
    sns.countplot(x=TARGET_COL, data=df)
    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "churn_distribution.png")
    plt.close()

    # =====================================================
    # 2️⃣ Tenure vs Churn
    # =====================================================
    if "tenure" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=TARGET_COL, y="tenure", data=df)
        plt.title("Tenure vs Churn")
        plt.xlabel("Churn")
        plt.ylabel("Tenure (months)")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "tenure_vs_churn.png")
        plt.close()

    # =====================================================
    # 3️⃣ Monthly Charges vs Churn
    # =====================================================
    if "MonthlyCharges" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=TARGET_COL, y="MonthlyCharges", data=df)
        plt.title("Monthly Charges vs Churn")
        plt.xlabel("Churn")
        plt.ylabel("Monthly Charges")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "monthly_charges_vs_churn.png")
        plt.close()

    # =====================================================
    # 4️⃣ Correlation Heatmap
    # =====================================================
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        corr = numeric_df.corr()
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            square=True,
            cbar=True,
        )
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "correlation_heatmap.png")
        plt.close()

    print("✅ EDA artifacts saved to artifacts/eda/")
