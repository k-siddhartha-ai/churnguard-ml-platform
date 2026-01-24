import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("artifacts/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def churn_distribution(df: pd.DataFrame):
    plt.figure()
    sns.countplot(x="Churn", data=df)
    plt.title("Customer Churn Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_distribution.png")
    plt.close()

def tenure_vs_churn(df: pd.DataFrame):
    plt.figure()
    sns.boxplot(x="Churn", y="tenure", data=df)
    plt.title("Tenure vs Churn")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tenure_vs_churn.png")
    plt.close()

def monthly_charges_vs_churn(df: pd.DataFrame):
    plt.figure()
    sns.violinplot(x="Churn", y="MonthlyCharges", data=df)
    plt.title("Monthly Charges vs Churn")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_charges_vs_churn.png")
    plt.close()

def correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()
