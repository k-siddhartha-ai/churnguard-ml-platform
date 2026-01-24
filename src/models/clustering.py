from sklearn.cluster import KMeans
import joblib

def train_kmeans(df):
    """
    Customer segmentation using KMeans.
    """
    X = df.drop("Churn", axis=1)

    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)

    joblib.dump(model, "artifacts/kmeans.pkl")
