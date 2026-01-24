# ChurnGuard  
### Real-Time Customer Churn Prediction & Customer Intelligence Platform

**Author:** K. Siddhartha  
**GitHub:** https://github.com/k-siddhartha-ai  
**LinkedIn:** http://www.linkedin.com/in/karne-siddhartha-163bb1369  

---

## 📌 Business Problem

Customer churn occurs when customers discontinue a company’s services.  
In subscription-driven businesses, acquiring a new customer costs **5–7× more** than retaining an existing one.

**ChurnGuard** predicts high-risk customers in advance using historical behavioral and billing data, enabling organizations to take **proactive, data-driven retention actions before revenue loss occurs**.

---

## 🎯 Project Objectives

- Predict customer churn using supervised machine learning  
- Identify churn-driving behavioral patterns  
- Segment customers using unsupervised learning  
- Build a **modular, production-style ML pipeline**  
- Deploy a **real-time inference web application**  

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**

Each record represents a customer with:
- Demographic information  
- Subscription & contract details  
- Service usage patterns  
- Billing and payment behavior  

**Target variable:** `Churn` (Yes / No)

---

## 🧠 End-to-End Machine Learning Pipeline  

### System Architecture

![ChurnGuard Architecture](docs/architecture.svg)

---

## 🔧 Training Pipeline (Offline)

The training pipeline is designed as a **reproducible, production-style workflow** to prevent data leakage and feature mismatch.

### Steps

1. **Data Ingestion**
   - Load raw Telco churn CSV data  
   - Handle encoding issues and malformed rows  

2. **Schema Validation**
   - Ensure presence of required features  
   - Separate target variable (`Churn`)  
   - Enforce correct data types  

3. **Data Cleaning**
   - Handle missing values  
   - Convert numeric fields safely (`TotalCharges`, `tenure`)  
   - Normalize categorical values  

4. **Feature Engineering**
   - ColumnTransformer-based preprocessing pipeline  
   - One-hot encoding for categorical features  
   - Scaling for numerical features  
   - Guarantees identical preprocessing during inference  

5. **Model Training**
   - Logistic Regression (baseline, interpretable)  
   - Random Forest (non-linear ensemble)  
   - KMeans clustering for customer segmentation  

6. **Model Evaluation**
   - ROC-AUC  
   - Precision, Recall, F1-score  
   - Metrics selected for churn class imbalance  

7. **Model Persistence**
   - Full pipelines serialized using `joblib`  
   - Stored as versioned artifacts for inference  

---

## 🌐 Inference Pipeline (Online / Real-Time)

The inference pipeline is built for **robust real-world usage**, handling unseen data safely.

### Steps

1. **User Upload**
   - Accept CSV or Excel files via Streamlit UI  

2. **Schema Alignment**
   - Automatically align uploaded data with training schema  
   - Missing categorical values filled as `"Unknown"`  
   - Missing numerical values defaulted to `0`  

3. **Preprocessing**
   - Apply the **same preprocessing pipeline used during training**  
   - Prevents feature mismatch and silent inference errors  

4. **Prediction**
   - Binary churn prediction (Yes / No)  
   - Churn probability scores  

5. **Risk Segmentation**
   - High Risk (≥ 70%)  
   - Medium Risk (40–69%)  
   - Low Risk (< 40%)  

6. **Explainability**
   - Logistic Regression coefficients  
   - Random Forest feature importance  
   - SHAP global feature impact  

7. **Output**
   - Interactive table in UI  
   - Downloadable business-ready CSV report  

---

## 🧪 Models Used

- **Logistic Regression** – Interpretable baseline model  
- **Random Forest Classifier** – Non-linear ensemble model  
- **KMeans Clustering** – Customer segmentation  

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  

Metrics are chosen to handle **class imbalance**, a common churn scenario.

---

## 🌐 Deployment

The application is deployed as a **Streamlit web app** with a FastAPI backend, supporting:

- Local execution  
- Hugging Face Spaces deployment  

Users can upload customer data and receive **real-time churn predictions**.

---

## 🛠 Tech Stack

- **Programming:** Python  
- **Data Processing:** Pandas, NumPy  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib, Seaborn  
- **Model Persistence:** Joblib  
- **Backend API:** FastAPI  
- **Web Application:** Streamlit  

---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

2️⃣ Start FastAPI backend
```bash
uvicorn main:app --reload
```
3️⃣ Launch the Streamlit frontend
```bash
streamlit run app/app.py
```
Open in browser:
http://localhost:8501

📁 Project Structure
churnguard/
├── config/          # Centralized configuration  
├── data/            # Raw & processed datasets  
├── docs/            # Architecture diagram  
├── src/             # Core ML modules  
├── pipelines/       # Training orchestration  
├── app/             # Streamlit inference app  
├── artifacts/       # Saved models & reports  
├── run.py           # Pipeline entry point  
├── requirements.txt  
└── README.md  

🚀 Results & Business Impact

Identifies high-risk churn customers before attrition

Enables targeted retention strategies

Reduces potential revenue loss

Demonstrates industry-grade ML system design, not just modeling


📬 Contact

If you find this project useful or would like to collaborate:

GitHub: https://github.com/k-siddhartha-ai

LinkedIn: http://www.linkedin.com/in/karne-siddhartha-163bb1369



