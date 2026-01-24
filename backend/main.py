from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd

# =====================================================
# Load models
# =====================================================
BASE_DIR = Path("../artifacts")

churn_model = joblib.load(BASE_DIR / "models/churn_pipeline.joblib")
sentiment_model = joblib.load(BASE_DIR / "nlp/sentiment_model.joblib")
vectorizer = joblib.load(BASE_DIR / "nlp/tfidf_vectorizer.joblib")

app = FastAPI(title="ChurnGuard ML API", version="1.0")

# =====================================================
# Request schemas
# =====================================================

from pydantic import BaseModel

class ChurnRequest(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class SentimentRequest(BaseModel):
    text: str

# =====================================================
# Routes
# =====================================================

@app.get("/")
def root():
    return {"message": "ChurnGuard API is running"}

# ---------- Churn Prediction API ----------
@app.post("/predict_churn")
def predict_churn(data: ChurnRequest):

    # Convert request JSON to dictionary
    input_dict = data.dict()

    # Convert to DataFrame (single row)
    df = pd.DataFrame([input_dict])

    # Predict using trained pipeline
    pred = churn_model.predict(df)[0]
    prob = churn_model.predict_proba(df)[0][1]

    return {
        "churn_prediction": "Yes" if pred == 1 else "No",
        "churn_probability": round(float(prob), 3)
    }


# ---------- Sentiment API ----------
@app.post("/predict_sentiment")
def predict_sentiment(data: SentimentRequest):

    X_vec = vectorizer.transform([data.text])
    probs = sentiment_model.predict_proba(X_vec)[0]

    idx = probs.argmax()
    label = sentiment_model.classes_[idx]
    confidence = float(probs[idx])

    return {
        "sentiment": label,
        "confidence": round(confidence, 3)
    }
