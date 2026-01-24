import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# =====================================================
# Load cleaned churn data
# =====================================================
DATA_PATH = Path("data/raw/telco_churn.csv")
df = pd.read_csv(DATA_PATH)

# Select important numerical features for segmentation
features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

df = df[features].copy()

# Handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.fillna(0)

# =====================================================
# Scale data
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# =====================================================
# Train KMeans
# =====================================================
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# =====================================================
# Save model & results
# =====================================================
ARTIFACTS_DIR = Path("artifacts/segmentation")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(kmeans, ARTIFACTS_DIR / "kmeans_model.joblib")
joblib.dump(scaler, ARTIFACTS_DIR / "kmeans_scaler.joblib")

df.to_csv(ARTIFACTS_DIR / "customer_segments.csv", index=False)

print("✅ KMeans model trained and segmentation saved.")
