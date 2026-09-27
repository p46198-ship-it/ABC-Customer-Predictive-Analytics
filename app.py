import streamlit as st
import pandas as pd
import joblib

# Load models
churn_model = joblib.load("abc_churn_model.pkl")
charges_model = joblib.load("abc_total_charges_model.pkl")

st.set_page_config(
    page_title="ABC Ltd. AI Predictive Tool",
    page_icon="📊",
    layout="wide"
)

st.title("ABC Ltd. – AI Customer Analytics Tool")
st.write("Predict customer churn probability and total customer charges.")

st.divider()

# Customer inputs
st.header("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", 0, 100, 12)

with col2:
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )
    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col3:
    device = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )
    support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

# Create input
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone],
    "MultipleLines": [multiple],
    "InternetService": [internet],
    "OnlineSecurity": [security],
    "OnlineBackup": [backup],
    "DeviceProtection": [device],
    "TechSupport": [support],
    "StreamingTV": [tv],
    "StreamingMovies": [movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless],
    "PaymentMethod": [payment],
    "MonthlyCharges": [monthly]
})

st.divider()

if st.button("🔮 Predict Customer Outcome", type="primary"):

    churn_probability = churn_model.predict_proba(input_data)[0][1]

    predicted_charges = charges_model.predict(input_data)[0]

    if churn_probability >= 0.70:
        risk = "HIGH"
    elif churn_probability >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Churn Probability",
        f"{churn_probability:.1%}"
    )

    col2.metric(
        "Churn Risk",
        risk
    )

    col3.metric(
        "Predicted Total Charges",
        f"{predicted_charges:,.2f}"
    )

    st.subheader("Managerial Insight")

    if risk == "HIGH":
        st.warning(
            "The customer has a relatively high predicted "
            "churn probability. Review the customer profile "
            "and consider appropriate retention action."
        )
    elif risk == "MEDIUM":
        st.info(
            "The customer has a moderate predicted churn "
            "probability and may require monitoring."
        )
    else:
        st.success(
            "The customer has a relatively low predicted "
            "churn probability."
        )

st.divider()

with st.expander("Model Performance"):
    st.write("### Logistic Regression – Churn Prediction")
    st.write("Accuracy: 72.71%")
    st.write("Precision: 49.16%")
    st.write("Recall: 78.34%")
    st.write("F1 Score: 60.41%")
    st.write("ROC-AUC: 83.30%")

    st.write("### Linear Regression – Total Charges")
    st.write("MAE: 541.02")
    st.write("RMSE: 673.95")
    st.write("R²: 0.9121")
