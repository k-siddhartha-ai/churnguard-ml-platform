from PIL import Image
import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from pathlib import Path
import requests
API_URL = "http://127.0.0.1:8000"


# =====================================================
# App Config
# =====================================================
st.set_page_config(page_title="ChurnGuard", layout="wide")

st.title("📉 ChurnGuard – Customer Churn Prediction")
st.caption("End-to-End ML System | Author: K. Siddhartha")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📂 Batch Prediction",
    "👤 Single Customer Prediction",
    "📊 EDA & Visualization",
    "📈 Model Comparison & Metrics",
    "🧠 Explainability & Feature Importance",
    "👥 Customer Segmentation (K-Means)",
    "🗣 Sentiment Analysis (NLP)",
    "🤖 Deep Learning Model"
])




# =====================================================
# Constants
# =====================================================
ARTIFACTS_DIR = Path("artifacts/models")
TARGET_COL = "Churn Label"

# =====================================================
# Load FULL PIPELINE MODEL
# =====================================================
@st.cache_resource
def load_model():
    model = joblib.load(ARTIFACTS_DIR / "churn_pipeline.joblib")
    return model

# =====================================================
# Robust File Reader
# =====================================================
def read_uploaded_file(uploaded):
    raw = uploaded.getvalue()
    try:
        if uploaded.name.lower().endswith(".xlsx"):
            return pd.read_excel(BytesIO(raw))
        return pd.read_csv(BytesIO(raw), encoding="latin1", engine="python")
    except Exception as e:
        st.error("❌ Could not read uploaded file")
        st.exception(e)
        return None

# =====================================================
# Schema Normalization
# =====================================================
def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    COLUMN_MAP = {
        "tenure": "Tenure in Months",
        "MonthlyCharges": "Monthly Charge",
        "SeniorCitizen": "Senior Citizen",
        "TotalCharges": "Total Charges",
    }

    df.rename(columns=COLUMN_MAP, inplace=True)

    for col in ["Tenure in Months", "Monthly Charge", "Total Charges"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# =====================================================
# TAB 1 — Batch Prediction
# =====================================================
with tab1:
    st.subheader("📂 Batch Customer Prediction")

    uploaded = st.file_uploader("Upload customer CSV or Excel", type=["csv", "xlsx"])

    if uploaded is None:
        st.info("📂 Please upload a CSV or Excel file to start prediction.")
    else:
        df = read_uploaded_file(uploaded)
        if df is None:
            st.stop()

        original_df = df.copy()
        df = normalize_schema(df)

        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])
            original_df = original_df.drop(columns=[TARGET_COL])

        st.success(f"File loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns)")
        st.subheader("🔍 Data Preview")
        st.dataframe(original_df.head(10), use_container_width=True)

        model = load_model()

        # Detect preprocessing
        preprocessor = None
        for name, step in model.named_steps.items():
            if hasattr(step, "transformers_"):
                preprocessor = step

        if preprocessor is None:
            st.error("❌ Could not find preprocessing step in pipeline")
            st.stop()

        df_model = df.copy()

        expected_cols = []
        for name, transformer, cols in preprocessor.transformers_:
            expected_cols.extend(list(cols))

        for col in expected_cols:
            if col not in df_model.columns:
                df_model[col] = None

        df_model = df_model[expected_cols]

        try:
            for name, transformer, cols in preprocessor.transformers_:

                # Numeric
                if "num" in name.lower():
                    for col in cols:
                        if col in df_model.columns:
                            df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
                            df_model[col] = df_model[col].fillna(0)

                # Categorical
                else:
                    if hasattr(transformer, "categories_"):
                        for i, col in enumerate(cols):
                            if col in df_model.columns:
                                known = transformer.categories_[i]
                                df_model[col] = df_model[col].astype(str)
                                df_model[col] = df_model[col].where(
                                    df_model[col].isin(known),
                                    other=known[0]
                                )

            preds = model.predict(df_model)
            probs = model.predict_proba(df_model)[:, 1]

        except Exception as e:
            st.error("❌ Model prediction failed")
            st.exception(e)
            st.stop()

        result = original_df.copy()
        result["Churn_Prediction"] = ["Yes" if x == 1 else "No" for x in preds]
        result["Churn_Probability (%)"] = (probs * 100).round(2)

        def risk_bucket(p):
            if p >= 70:
                return "High Risk 🔴"
            elif p >= 40:
                return "Medium Risk 🟠"
            return "Low Risk 🟢"

        result["Risk_Level"] = result["Churn_Probability (%)"].apply(risk_bucket)

        churn_rate = round(probs.mean() * 100, 2)
        st.metric("Predicted Churn Rate (%)", f"{churn_rate}%")

        st.subheader("🤖 Model Predictions")
        st.dataframe(result.head(20), use_container_width=True)

        st.download_button(
            "⬇️ Download Predictions",
            data=result.to_csv(index=False),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

# =====================================================
# TAB 2 — Single Customer
# =====================================================
# =====================================================
# TAB 2 — Single Customer (API MODE – INDUSTRY STYLE)
# =====================================================
with tab2:
    st.subheader("👤 Single Customer Prediction (via FastAPI)")

    st.markdown("### Enter customer details")

    customerID = st.text_input("Customer ID", "0001-BGHTR")
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])

    MonthlyCharges = st.number_input("Monthly Charges", value=75.5)
    TotalCharges = st.number_input("Total Charges", value=850.3)

    # -------------------------
    # CALL FASTAPI
    # -------------------------
    if st.button("🔮 Predict Churn (via API)"):

        payload = {
            "customerID": customerID,
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges
        }

        try:
            response = requests.post(
                f"{API_URL}/predict_churn",
                json=payload
            )

            if response.status_code == 200:
                result = response.json()

                pred = result["churn_prediction"]
                prob = result["churn_probability"] * 100

                st.markdown("### 📊 Prediction Result")

                if pred == "Yes":
                    st.error(f"⚠️ Customer WILL churn")
                else:
                    st.success(f"✅ Customer will NOT churn")

                st.metric("Churn Risk (%)", f"{prob:.2f}%")

            else:
                st.error("❌ API Error")
                st.text(response.text)

        except Exception as e:
            st.error("❌ Could not connect to FastAPI server")
            st.exception(e)

# =====================================================
# TAB 3 — EDA & Visualization
# =====================================================
with tab3:
    st.subheader("📊 Exploratory Data Analysis (EDA) & Visualizations")

    EDA_DIR = Path("artifacts/eda")

    st.markdown("### 🔍 Key Business Insights from Data")

    def show_plot(title, filename):
        path = EDA_DIR / filename
        if path.exists():
            st.markdown(f"#### {title}")
            img = Image.open(path)

            st.image(img, width=600)

            with st.expander("🔍 Click to enlarge"):
                st.image(img, use_container_width=True)

            with open(path, "rb") as f:
                st.download_button(
                    label="⬇️ Download this plot",
                    data=f,
                    file_name=filename,
                    mime="image/png"
                )
        else:
            st.warning(f"⚠️ Plot not found: {filename}")

    col1, col2 = st.columns(2)
    with col1:
        show_plot("Churn Distribution", "churn_distribution.png")
    with col2:
        show_plot("Correlation Heatmap", "correlation_heatmap.png")

    col3, col4 = st.columns(2)
    with col3:
        show_plot("Tenure vs Churn", "tenure_vs_churn.png")
    with col4:
        show_plot("Monthly Charges vs Churn", "monthly_charges_vs_churn.png")

    st.success("✅ EDA loaded successfully from training artifacts.")

# =====================================================
# TAB 4 — Model Comparison & Metrics
# =====================================================
with tab4:
    st.subheader("📈 Model Comparison & Evaluation Dashboard")

    REPORTS_DIR = Path("artifacts/reports")

    metrics_file = REPORTS_DIR / "model_comparison_metrics.csv"

    if metrics_file.exists():
        st.markdown("### 🏆 Model Performance Comparison")
        df_metrics = pd.read_csv(metrics_file)
        st.dataframe(df_metrics, use_container_width=True)
    else:
        st.warning("⚠️ model_comparison_metrics.csv not found")

    roc_path = REPORTS_DIR / "churn_pipeline_roc_curve.png"
    if roc_path.exists():
        st.markdown("### 📊 ROC Curve (Best Model)")
        img = Image.open(roc_path)
        st.image(img, width=600)

        with st.expander("🔍 Enlarge ROC Curve"):
            st.image(img, use_container_width=True)
    else:
        st.warning("⚠️ ROC curve image not found")

    cm_path = REPORTS_DIR / "churn_pipeline_confusion_matrix.png"
    if cm_path.exists():
        st.markdown("### 🧮 Confusion Matrix")
        img = Image.open(cm_path)
        st.image(img, width=600)

        with st.expander("🔍 Enlarge Confusion Matrix"):
            st.image(img, use_container_width=True)
    else:
        st.warning("⚠️ Confusion matrix image not found")

    st.markdown("""
    ### 📌 What this shows recruiters:
    - Multiple model training (Logistic, Random Forest)  
    - Model comparison & selection  
    - ROC–AUC analysis  
    - Confusion matrix interpretation  
    - Industry-standard ML evaluation  
    """)
# =====================================================
# TAB 5 — Explainability & Feature Importance (XAI)
# =====================================================
with tab5:
    st.subheader("🧠 Model Explainability & Feature Importance")

    EXPLAIN_DIR = Path("artifacts/explainability")

    # ---------------------------
    # Logistic Regression Coefficients
    # ---------------------------
    lr_path = EXPLAIN_DIR / "logistic_coefficients.csv"

    if lr_path.exists():
        st.markdown("### 📌 Logistic Regression — Feature Coefficients")

        df_lr = pd.read_csv(lr_path)
        df_lr = df_lr.sort_values(by="coefficient", ascending=False)

        st.dataframe(df_lr.head(30), use_container_width=True)

        st.markdown("""
        🔍 Interpretation:
        - Positive coefficient → increases churn probability  
        - Negative coefficient → decreases churn probability  
        - Higher absolute value → more important feature  
        """)

    else:
        st.warning("⚠️ logistic_coefficients.csv not found")

    # ---------------------------
    # Random Forest Feature Importance
    # ---------------------------
    rf_path = EXPLAIN_DIR / "random_forest_importance.csv"

    if rf_path.exists():
        st.markdown("### 🌲 Random Forest — Feature Importance")

        df_rf = pd.read_csv(rf_path)
        df_rf = df_rf.sort_values(by="importance", ascending=False)

        st.dataframe(df_rf.head(30), use_container_width=True)

        st.markdown("""
        🔍 Interpretation:
        - Higher importance → stronger influence on predictions  
        - Helps identify key churn drivers  
        - Used in business decision making  
        """)

    else:
        st.warning("⚠️ random_forest_importance.csv not found")

    st.success("✅ Explainability artifacts loaded successfully.")

    st.markdown("""
    ### 📌 What this shows recruiters:
    - Model interpretability (Explainable AI)  
    - Feature contribution analysis  
    - Business insight extraction  
    - Industry-standard compliance & trust  
    """)
# =====================================================
# TAB 6 — Customer Segmentation (Unsupervised Learning)
# =====================================================
with tab6:
    st.subheader("👥 Customer Segmentation using K-Means Clustering")

    SEG_DIR = Path("artifacts/segmentation")

    # ---------------------------
    # Load segmented customers
    # ---------------------------
    seg_file = SEG_DIR / "customer_segments.csv"

    if seg_file.exists():
        df_seg = pd.read_csv(seg_file)

        st.markdown("### 🧾 Sample Customer Segments")
        st.dataframe(df_seg.head(20), use_container_width=True)

        st.markdown("### 📊 Cluster Distribution")
        cluster_counts = df_seg["Cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

    else:
        st.warning("⚠️ customer_segments.csv not found")

    # ---------------------------
    # Show cluster plots
    # ---------------------------
    plot1 = SEG_DIR / "clusters_tenure_monthly.png"
    plot2 = SEG_DIR / "clusters_total_monthly.png"

    col1, col2 = st.columns(2)

    with col1:
        if plot1.exists():
            st.markdown("### Tenure vs Monthly Charges")
            img = Image.open(plot1)
            st.image(img, width=500)
        else:
            st.warning("⚠️ clusters_tenure_monthly.png not found")

    with col2:
        if plot2.exists():
            st.markdown("### Total Charges vs Monthly Charges")
            img = Image.open(plot2)
            st.image(img, width=500)
        else:
            st.warning("⚠️ clusters_total_monthly.png not found")

    st.markdown("""
    ### 📌 What this shows recruiters:
    - Unsupervised learning (K-Means clustering)  
    - Customer segmentation & profiling  
    - Distance-based learning (Euclidean)  
    - Business-oriented analytics  
    - Industry-standard customer intelligence  
    """)

# =====================================================
# TAB 7 — Sentiment Analysis (NLP Module)
# =====================================================
with tab7:
    st.subheader("🗣 Customer Review Sentiment Analysis (NLP)")

    NLP_DIR = Path("artifacts/nlp")

    model_path = NLP_DIR / "sentiment_model.joblib"
    vec_path = NLP_DIR / "tfidf_vectorizer.joblib"

    # Load your trained English model
    if not model_path.exists() or not vec_path.exists():
        st.error("❌ Sentiment model or vectorizer not found. Please train NLP model first.")
        st.stop()

    sentiment_model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    # Load multilingual industry model (BERT)
    from transformers import pipeline
    from langdetect import detect


    @st.cache_resource
    def load_multilang_model():
        # Force PyTorch backend (avoids Keras / TensorFlow issues)
        return pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            framework="pt"  # 🔥 THIS FIXES THE ERROR
        )


    multilang_model = load_multilang_model()

    # ---------------------------
    # UI
    # ---------------------------
    st.markdown("### ✍️ Enter a customer review (any language)")

    mode = st.radio(
        "Select Sentiment Engine",
        ["English ML Model (Your Training)", "Multi-Language Industry Model"],
        horizontal=True
    )

    user_text = st.text_area("Customer Review", height=120)

    if st.button("🔍 Analyze Sentiment"):

        if user_text.strip() == "":
            st.warning("⚠️ Please enter some text.")
            st.stop()

        # =====================================================
        # OPTION 1 — YOUR ENGLISH ML MODEL
        # =====================================================
        if mode == "English ML Model (Your Training)":

            X_vec = vectorizer.transform([user_text])
            probs = sentiment_model.predict_proba(X_vec)[0]

            pred_idx = probs.argmax()
            confidence = probs[pred_idx]
            label = sentiment_model.classes_[pred_idx]

            # Low confidence handling
            st.write(f"🧪 Model confidence: {confidence:.2f}")
            st.write("🔹 Engine: Logistic Regression + TF-IDF (Your trained model)")

            if confidence < 0.60:
                st.info(f"😐 Sentiment: NEUTRAL (low confidence {confidence:.2f})")
            else:
                if label == "positive":
                    st.success(f"😊 Sentiment: POSITIVE ({confidence:.2f})")
                elif label == "negative":
                    st.error(f"😠 Sentiment: NEGATIVE ({confidence:.2f})")


        # =====================================================
        # OPTION 2 — MULTI-LANGUAGE INDUSTRY MODEL
        # =====================================================
        else:
            try:
                lang = detect(user_text)
            except:
                lang = "unknown"

            result = multilang_model(user_text)[0]
            label = result["label"]
            score = result["score"]

            st.write(f"🌍 Detected language: {lang}")
            st.write(f"🧪 Model confidence: {score:.2f}")

            # Convert star rating to sentiment
            if "1 star" in label or "2 star" in label:
                st.error(f"😠 Sentiment: NEGATIVE ({score:.2f})")
            elif "3 star" in label:
                st.info(f"😐 Sentiment: NEUTRAL ({score:.2f})")
            else:
                st.success(f"😊 Sentiment: POSITIVE ({score:.2f})")

            st.write("🔹 Engine: Multilingual BERT (Industry pretrained model)")

    st.markdown("""
    ### 📌 What this shows recruiters:
    - Classical ML (TF-IDF + Logistic Regression)  
    - Transformer-based NLP (BERT, multilingual)  
    - Language detection & internationalization  
    - Real-time sentiment inference  
    - Production-style hybrid NLP system  
    """)

# =====================================================
# TAB 8 — Deep Learning Churn Prediction (Neural Network)
# =====================================================
with tab8:
    st.subheader("🤖 Churn Prediction using Deep Learning (Neural Network)")

    DL_DIR = Path("artifacts/deep_learning")

    model_path = DL_DIR / "churn_nn_model.h5"
    scaler_path = DL_DIR / "nn_scaler.joblib"
    feat_path = DL_DIR / "nn_features.joblib"

    if not model_path.exists():
        st.error("❌ Neural network model not found. Please train deep learning model first.")
        st.stop()

    # Load model & artifacts
    @st.cache_resource
    def load_dl_artifacts():
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(feat_path)
        return model, scaler, features


    nn_model, nn_scaler, nn_features = load_dl_artifacts()

    st.markdown("### 📂 Upload customer file for Deep Learning prediction")

    uploaded_dl = st.file_uploader("Upload CSV (same schema as churn data)", type=["csv"], key="dl_upload")

    if uploaded_dl:
        df_original = pd.read_csv(uploaded_dl)

        df_dl = df_original.copy()

        if "Churn" in df_dl.columns:
            df_dl = df_dl.drop(columns=["Churn"])

        # Encode categorical
        df_encoded = pd.get_dummies(df_dl, drop_first=True)

        # Align features with training
        for col in nn_features:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded[nn_features]

        df_encoded = df_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Scale
        X_scaled = nn_scaler.transform(df_encoded)

        # Predict
        probs = nn_model.predict(X_scaled).ravel()
        preds = (probs > 0.5).astype(int)

        # FINAL RESULT (BUSINESS FRIENDLY)
        result = df_original.copy()

        result["Churn_Prediction_DL"] = ["Yes" if x == 1 else "No" for x in preds]
        result["Churn_Probability_DL (%)"] = (probs * 100).round(2)


        def risk_bucket(p):
            if p >= 70:
                return "High Risk 🔴"
            elif p >= 40:
                return "Medium Risk 🟠"
            return "Low Risk 🟢"


        result["Risk_Level_DL"] = result["Churn_Probability_DL (%)"].apply(risk_bucket)

        st.success("✅ Deep Learning predictions completed")

        st.dataframe(result.head(20), use_container_width=True)

        st.download_button(
            "⬇️ Download Deep Learning Predictions",
            data=result.to_csv(index=False),
            file_name="deep_learning_churn_predictions.csv",
            mime="text/csv",
        )
    st.markdown("""
    ### 🔍 How to interpret Deep Learning output:
    - Probability ≥ 70% → Very likely to churn  
    - 40–70% → At risk (monitor / retention offer)  
    - < 40% → Low churn risk  
    """)
