import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import datetime


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)


# Custom Bank-Style CSS

st.markdown("""
<style>

/* App background */
.stApp {
    background: linear-gradient(135deg, #f4f8fb, #eaf1f8);
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #0a2a43;
}

/* Sidebar text */
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] h2 {
    color: white !important;
}

/* Title */
h1 {
    color: #0a2a43;
    font-weight: 700;
}

/* Cards */
.card {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #004e92, #000428);
    color: white;
    border-radius: 12px;
    padding: 12px 26px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease-in-out;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #000428, #004e92);
    transform: scale(1.02);
}

/* Metric */
[data-testid="stMetricValue"] {
    color: #004e92;
    font-size: 32px;
    font-weight: 700;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background-color: #004e92;
}

/* Alerts */
.stAlert {
    border-radius: 12px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# Load Model & Preprocessors

model = tf.keras.models.load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    label_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Title Section

st.markdown("""
<h1 style="text-align:center;">🏦 Customer Churn Prediction</h1>
<p style="text-align:center; font-size:18px;">
AI-powered system to predict whether a customer is likely to leave the bank
</p>
<hr>
""", unsafe_allow_html=True)

# Sidebar Inputs

st.sidebar.header("🧾 Customer Information")

geography = st.sidebar.selectbox(
    "🌍 Geography",
    onehot_encoder_geo.categories_[0]
)

gender = st.sidebar.selectbox(
    "👤 Gender",
    label_encoder_gender.classes_
)

age = st.sidebar.slider("🎂 Age", 18, 92)
tenure = st.sidebar.slider("📆 Tenure (Years)", 0, 10)

credit_score = st.sidebar.number_input(
    "💳 Credit Score", min_value=300, max_value=900
)

balance = st.sidebar.number_input(
    "🏦 Account Balance", min_value=0.0
)

estimated_salary = st.sidebar.number_input(
    "💰 Estimated Salary", min_value=0.0
)

num_of_products = st.sidebar.slider(
    "📦 Number of Products", 1, 4
)

has_cr_card = st.sidebar.selectbox(
    "💳 Has Credit Card", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

is_active_member = st.sidebar.selectbox(
    "⚡ Active Member", [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)


# Main Content


st.subheader("📌 Predict Customer Churn")
st.write(
    "Fill in the customer details from the sidebar and click **Predict Churn** "
    "to estimate the probability of customer churn."
)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction

if st.button("🚀 Predict Churn", use_container_width=True):

    with st.spinner("Analyzing customer data..."):

        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [label_encoder_gender.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })

        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
        )

        final_input = pd.concat(
            [input_data.reset_index(drop=True), geo_df],
            axis=1
        )

        final_input_scaled = scaler.transform(final_input)

        prediction = model.predict(final_input_scaled)
        churn_prob = prediction[0][0]

    # Result Display
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Prediction Result")

    st.progress(int(churn_prob * 100))
    st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")

    if churn_prob > 0.5:
        st.error("⚠️ High Risk: The customer is likely to churn.")
    else:
        st.success("✅ Low Risk: The customer is likely to stay.")

    st.markdown('</div>', unsafe_allow_html=True)
