
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fraud_detection_pipeline.pkl")

# Page config
st.set_page_config(page_title="Fraud Detection", layout="centered")

# Header
st.title("💳 Fraud Detection Dashboard")
st.markdown("Analyze transactions and detect potential fraud in real time")

st.divider()

# Layout in 2 columns
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "DEPOSIT"]
    )
    amount = st.number_input("Amount", min_value=0.0, value=1000.0)
    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=10000.0)

with col2:
    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=9000.0)
    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=0.0)

st.divider()

# Prediction button
if st.button("🔍 Analyze Transaction"):

    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Analysis Result")

    # Metric display
    st.metric(label="Fraud Probability", value=f"{probability:.2%}")

    # Progress bar
    st.progress(float(probability))

    # Result message
    if prediction == 1:
        st.error("⚠️ High Risk: Fraudulent Transaction Detected")
    else:
        st.success("✅ Low Risk: Legitimate Transaction")

    # Interpretation
    st.markdown("### 🧠 Interpretation")
    if probability > 0.8:
        st.write("Very high likelihood of fraud. Immediate action recommended.")
    elif probability > 0.5:
        st.write("Moderate risk. Review transaction carefully.")
    else:
        st.write("Low risk. Transaction appears normal.")
