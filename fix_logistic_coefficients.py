import pandas as pd

# Load coefficients
df = pd.read_csv("artifacts/explainability/logistic_coefficients.csv")

# 🔴 REMOVE customerID FEATURES (they are meaningless)
df = df[~df["feature"].str.contains("customerID", na=False)]

# 🔹 Compute proper impact = absolute coefficient
df["impact"] = df["coefficient"].abs()

# 🔹 Sort by impact
df = df.sort_values(by="impact", ascending=False)

# 🔹 Keep only top 30 meaningful features
df = df.head(30)

# Save back
df.to_csv("artifacts/explainability/logistic_coefficients.csv", index=False)

print("✅ Logistic explainability fixed (customerID removed, impact corrected)")
