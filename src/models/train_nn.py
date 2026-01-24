import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# Load cleaned churn data
# =====================================================
df = pd.read_csv("data/raw/telco_churn.csv")

# Target
y = df["Churn"].map({"Yes": 1, "No": 0})

# Drop target
X = df.drop(columns=["Churn"])

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Handle missing values
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

# =====================================================
# Train-test split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =====================================================
# Scale features
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# Build Neural Network
# =====================================================
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =====================================================
# Train model
# =====================================================
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# =====================================================
# Evaluate
# =====================================================
y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).ravel()

acc = accuracy_score(y_test, y_pred)
print("\nNeural Network Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =====================================================
# Save model & scaler
# =====================================================
ARTIFACTS_DIR = Path("artifacts/deep_learning")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

model.save(ARTIFACTS_DIR / "churn_nn_model.h5")
joblib.dump(scaler, ARTIFACTS_DIR / "nn_scaler.joblib")
joblib.dump(X_train.columns.tolist(), ARTIFACTS_DIR / "nn_features.joblib")

print("✅ Neural network model and artifacts saved.")
