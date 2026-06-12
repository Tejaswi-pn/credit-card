import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# ==================================
# LOAD MODEL
# ==================================
try:
    model = joblib.load("credit_card_default_model (2).pkl")
    preprocessor = joblib.load("preprocessor (2).pkl")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# ==================================
# HEADER
# ==================================
st.title("💳 Credit Card Default Risk Prediction")
st.markdown("Predict whether a customer is likely to default on their credit card payment.")

# ==================================
# SIDEBAR INPUTS
# ==================================
st.sidebar.header("Customer Information")

limit_bal = st.sidebar.number_input(
    "Credit Limit",
    min_value=10000,
    value=200000
)

sex = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

education = st.sidebar.selectbox(
    "Education",
    ["Graduate School", "University", "High School", "Others"]
)

marriage = st.sidebar.selectbox(
    "Marriage",
    ["Married", "Single", "Others"]
)

age = st.sidebar.slider(
    "Age",
    21,
    80,
    35
)

st.sidebar.subheader("Repayment Status")

pay_0 = st.sidebar.slider("PAY_0", -2, 8, 0)
pay_2 = st.sidebar.slider("PAY_2", -2, 8, 0)
pay_3 = st.sidebar.slider("PAY_3", -2, 8, 0)
pay_4 = st.sidebar.slider("PAY_4", -2, 8, 0)
pay_5 = st.sidebar.slider("PAY_5", -2, 8, 0)
pay_6 = st.sidebar.slider("PAY_6", -2, 8, 0)

st.sidebar.subheader("Bill Amounts")

bill_amt1 = st.sidebar.number_input("BILL_AMT1", value=5000)
bill_amt2 = st.sidebar.number_input("BILL_AMT2", value=5000)
bill_amt3 = st.sidebar.number_input("BILL_AMT3", value=5000)
bill_amt4 = st.sidebar.number_input("BILL_AMT4", value=5000)
bill_amt5 = st.sidebar.number_input("BILL_AMT5", value=5000)
bill_amt6 = st.sidebar.number_input("BILL_AMT6", value=5000)

st.sidebar.subheader("Payment Amounts")

pay_amt1 = st.sidebar.number_input("PAY_AMT1", value=2000)
pay_amt2 = st.sidebar.number_input("PAY_AMT2", value=2000)
pay_amt3 = st.sidebar.number_input("PAY_AMT3", value=2000)
pay_amt4 = st.sidebar.number_input("PAY_AMT4", value=2000)
pay_amt5 = st.sidebar.number_input("PAY_AMT5", value=2000)
pay_amt6 = st.sidebar.number_input("PAY_AMT6", value=2000)

predict_btn = st.sidebar.button("Predict")

# ==================================
# PREDICTION
# ==================================
if predict_btn:

    try:

        input_df = pd.DataFrame({
            "limit_bal": [limit_bal],
            "sex": [sex],
            "education": [education],
            "marriage": [marriage],
            "age": [age],

            "pay_0": [pay_0],
            "pay_2": [pay_2],
            "pay_3": [pay_3],
            "pay_4": [pay_4],
            "pay_5": [pay_5],
            "pay_6": [pay_6],

            "bill_amt1": [bill_amt1],
            "bill_amt2": [bill_amt2],
            "bill_amt3": [bill_amt3],
            "bill_amt4": [bill_amt4],
            "bill_amt5": [bill_amt5],
            "bill_amt6": [bill_amt6],

            "pay_amt1": [pay_amt1],
            "pay_amt2": [pay_amt2],
            "pay_amt3": [pay_amt3],
            "pay_amt4": [pay_amt4],
            "pay_amt5": [pay_amt5],
            "pay_amt6": [pay_amt6]
        })

        X = preprocessor.transform(input_df)

        prediction = model.predict(X)[0]

        probability = model.predict_proba(X)[0]

        no_default_prob = probability[0] * 100
        default_prob = probability[1] * 100

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Prediction")

            if prediction == "Y":
                st.error(
                    f"⚠ Customer likely to DEFAULT\n\nRisk Score: {default_prob:.2f}%"
                )
            else:
                st.success(
                    f"✅ Customer NOT likely to DEFAULT\n\nConfidence: {no_default_prob:.2f}%"
                )

        with col2:

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=default_prob,
                title={"text": "Default Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "steps": [
                        {"range": [0, 40], "color": "lightgreen"},
                        {"range": [40, 70], "color": "orange"},
                        {"range": [70, 100], "color": "red"},
                    ],
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Probability Distribution")

        bar_fig = go.Figure()

        bar_fig.add_bar(
            x=["No Default", "Default"],
            y=[no_default_prob, default_prob]
        )

        st.plotly_chart(bar_fig, use_container_width=True)

        st.subheader("Input Data")

        st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

        st.write("Expected Columns:")
        try:
            st.write(preprocessor.feature_names_in_)
        except:
            pass

else:
    st.info("Enter details in the sidebar and click Predict.")