import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load segmented data
df = pd.read_csv("artifacts/segmentation/customer_segments.csv")

PLOT_DIR = Path("artifacts/segmentation")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Plot 1 — Tenure vs Monthly Charges
plt.figure(figsize=(8, 6))
plt.scatter(df["tenure"], df["MonthlyCharges"], c=df["Cluster"], cmap="viridis", alpha=0.6)
plt.xlabel("Tenure")
plt.ylabel("Monthly Charges")
plt.title("Customer Segments (Tenure vs Monthly Charges)")
plt.colorbar(label="Cluster")
plt.savefig(PLOT_DIR / "clusters_tenure_monthly.png")
plt.close()

# Plot 2 — Total Charges vs Monthly Charges
plt.figure(figsize=(8, 6))
plt.scatter(df["TotalCharges"], df["MonthlyCharges"], c=df["Cluster"], cmap="plasma", alpha=0.6)
plt.xlabel("Total Charges")
plt.ylabel("Monthly Charges")
plt.title("Customer Segments (Total vs Monthly Charges)")
plt.colorbar(label="Cluster")
plt.savefig(PLOT_DIR / "clusters_total_monthly.png")
plt.close()

print("✅ Cluster plots saved.")
