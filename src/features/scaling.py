from sklearn.preprocessing import StandardScaler

def scale_features(X):
    """
    Scale numerical features.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(X)
