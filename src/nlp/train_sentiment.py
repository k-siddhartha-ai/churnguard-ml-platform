import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# =====================================================
# Better training dataset (you can extend this later)
# =====================================================
data = {
    "review": [
        "excellent service", "very good experience", "happy with support",
        "great customer care", "awesome internet speed",

        "bad service", "very poor experience", "terrible support",
        "worst network", "not satisfied with service",

        "not bad experience", "not good service", "not very happy",
        "not satisfied", "not excellent",

        "service was not good", "support is not helpful",
        "connection is not bad", "billing is not correct", "network is very bad"
    ],
    "sentiment": [
        "positive", "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative", "negative",
        "positive", "negative", "negative", "negative", "negative",
        "negative", "negative", "positive", "negative", "negative"
    ]
}

df = pd.DataFrame(data)

# =====================================================
# Train-test split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.25, random_state=42
)

# =====================================================
# Vectorization (important: bigrams for "not good", "not bad")
# =====================================================
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),      # handles "not good", "very bad"
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================================================
# Train model (Logistic Regression is stronger than Naive Bayes)
# =====================================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# =====================================================
# Evaluate
# =====================================================
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# =====================================================
# Save model & vectorizer
# =====================================================
ARTIFACTS_DIR = Path("artifacts/nlp")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, ARTIFACTS_DIR / "sentiment_model.joblib")
joblib.dump(vectorizer, ARTIFACTS_DIR / "tfidf_vectorizer.joblib")

print("✅ Improved sentiment model and vectorizer saved successfully.")
