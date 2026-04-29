
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn

# Load model
with open("final_loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.markdown(
    "<h1 style='text-align: center; background-color: #000080; padding: 10px; color: #F0FFFF;'><b>Your Loan Approval</b></h1>",
    unsafe_allow_html=True
)

st.image("https://cdn-icons-gif.flaticon.com/19005/19005108.gif")
st.header("Enter Loan Details:")

# --- INPUTS ---
reason = st.selectbox(
    "Reason for Loan",
    ['Cover An Unexpected Cost', 'Credit Card Refinancing', 'Home Improvement',
     'Major Purchase', 'Debt Consolidation', 'Other']  # ✅ FIXED TYPO
)

requestloanamt = st.slider("Loan Amount", 1000, 50000, step=100)
FICOscore = st.slider("FICO Score", 0, 850, step=1)

EverBankruptOrForeclose = st.checkbox("Ever Bankrupt or Foreclosed")

MonthlyHousingPayment = st.number_input(
    "Monthly Housing Payment", min_value=300, max_value=49500, step=1
)

Lender = st.pills("Lender", ["A", "B", "C"])
EmploymentStatus = st.pills("Employment Status", ["Full Time", "Part Time", "Unemployed"])

MonthlyGrossIncome = st.number_input(
    "Monthly Gross Income", min_value=0, max_value=100000, step=1
)

# --- HANDLE NONE VALUES FROM PILLS ---
if Lender is None:
    Lender = "A"   # default
if EmploymentStatus is None:
    EmploymentStatus = "Full Time"  # default

# --- CREATE INPUT DATA ---
input_data = pd.DataFrame({
    "Reason": [reason],
    "Requested_Loan_Amount": [requestloanamt],
    "FICO_score": [FICOscore],
    "Ever_Bankrupt_or_Foreclosed": [int(EverBankruptOrForeclose)],  # ✅ FIX
    "Monthly_Housing_Payment": [MonthlyHousingPayment],
    "Lender": [Lender],
    "Employment_Status": [EmploymentStatus],
    "Monthly_Gross_Income": [MonthlyGrossIncome]
})

# --- ENCODE (MATCH TRAINING) ---
input_data_encoded = pd.get_dummies(
    input_data,
    columns=['Reason', 'Lender', 'Employment_Status'],
    drop_first=True   # ✅ MUST MATCH TRAINING
)

# --- ALIGN COLUMNS SAFELY ---
model_columns = model.feature_names_in_

input_data_encoded = input_data_encoded.reindex(
    columns=model_columns,
    fill_value=0
)

# --- PREDICTION ---
if st.button("Evaluate Loan"):

    prediction = model.predict(input_data_encoded)[0]
    probs = model.predict_proba(input_data_encoded)[0]

    # --- DEBUG (you can remove later) ---
    st.write("Encoded Input:", input_data_encoded)
    st.write("Prediction:", prediction)
    st.write("Probabilities:", probs)

    # --- OUTPUT ---
    if prediction == 1:
        st.success("Your Loan is Approved!")
    else:
        st.error("Your Loan is Denied.")

st.image("https://static.vecteezy.com/system/resources/thumbnails/000/287/135/small/1__284_29.jpg")
