# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# LOAD MODEL
# -----------------------------
with open("final_loan_model.pkl", "rb") as file:
    model = pickle.load(file)
model_columns = pickle.load(open("model_columns.pkl", "rb"))
# -----------------------------
# PAGE HEADER
# -----------------------------
st.markdown(
    "<h1 style='text-align: center; background-color: #000080; padding: 10px; color: #F0FFFF;'><b>Your Loan Approval</b></h1>",
    unsafe_allow_html=True
)

st.image("https://cdn-icons-gif.flaticon.com/19005/19005108.gif")
st.header("Enter Loan Details:")

# -----------------------------
# CLEANING FUNCTION (CRITICAL FIX)
# -----------------------------
def clean(x):
    return str(x).strip().lower().replace(" ", "_")

# -----------------------------
# INPUTS
# -----------------------------
reason = st.selectbox("Reason", [
    "cover_an_unexpected_cost",
    "credit_card_refinancing",
    "home_improvement",
    "major_purchase",
    "debt_conslidation",
    "other"
])

requestloanamt = st.slider("Loan Amount", 1000, 50000, step=100)
FICOscore = st.slider("FICO Score", 0, 850, step=1)
EverBankruptOrForeclose = st.checkbox("Ever Bankrupt or Foreclosed")
MonthlyHousingPayment = st.number_input("Monthly Housing Payment", min_value=300, max_value=49500, step=1)

lender = st.selectbox("Lender", ["A", "B", "C"])

employment_status = st.selectbox("Employment Status", [
    "full_time",
    "part_time",
    "unemployed"
])

employment_sector = st.selectbox("Employment Sector", [
    "consumer_discretionary",
    "information_technology",
    "energy",
    "consumer_staples",
    "communication_services",
    "materials",
    "utilities",
    "health_care",
    "real_estate",
    "financials",
    "industrials"
])

MonthlyGrossIncome = st.number_input("Monthly Gross Income", 0, 100000, step=1)

# -----------------------------
# FICO GROUP FUNCTION
# -----------------------------
def fico_group(score):
    if score >= 800:
        return "excellent"
    elif score >= 740:
        return "very_good"
    elif score >= 670:
        return "good"
    elif score >= 580:
        return "fair"
    else:
        return "poor"

# -----------------------------
# BUILD INPUT DATA (CLEANED FIRST)
# -----------------------------
input_data = pd.DataFrame([{
    "Reason": clean(reason),
    "Requested_Loan_Amount": requestloanamt,
    "FICO_score": FICOscore,
    "Fico_Score_group": fico_group(FICOscore),
    "Employment_Status": clean(employment_status),
    "Employment_Sector": clean(employment_sector),
    "Monthly_Gross_Income": MonthlyGrossIncome,
    "Monthly_Housing_Payment": MonthlyHousingPayment,
    "Ever_Bankrupt_or_Foreclose": int(EverBankruptOrForeclose),
    "Lender": clean(lender)
}])

input_data_encoded = pd.get_dummies(input_data, drop_first=True)

# FORCE EXACT TRAINING COLUMNS
input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Evaluate Loan"):

    prediction = model.predict(input_data_encoded)[0]
    probs = model.predict_proba(input_data_encoded)[0]
    leaf = model.apply(input_data_encoded)[0]

    st.subheader("Results")

    st.write(f"Approval Probability: {probs[1]:.2%}")
    st.write("Leaf Node:", leaf)

    # Debug: show only active features
    st.write("Active Features:")
    st.write(input_data_encoded.loc[:, input_data_encoded.sum() > 0])

    if prediction == 1:
        st.success("Your Loan is Approved!")
    else:
        st.error("Your Loan is Denied.")

st.write(model.feature_names_in_)
st.write(input_data_encoded.sum().sort_values(ascending=False))
st.write("Categorical sum check:")
st.write(input_data_encoded.filter(like="_").sum().sort_values())
st.write(input_data_encoded.sum().sort_values())
st.write(input_data_encoded.filter(like="_").sum())

# -----------------------------
# FOOTER IMAGE
# -----------------------------
st.image("https://static.vecteezy.com/system/resources/thumbnails/000/287/135/small/1__284_29.jpg")
