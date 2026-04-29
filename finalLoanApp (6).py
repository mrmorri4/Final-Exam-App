
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("final_loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
#st.title("Your Loan Approval")
st.markdown(
    "<h1 style='text-align: center; background-color: #000080; padding: 10px; color: #F0FFFF;'><b>Your Loan Approval</b></h1>",
    unsafe_allow_html=True
)

st.image("https://cdn-icons-gif.flaticon.com/19005/19005108.gif")
st.header("Enter Loan Details:")

reason = st.selectbox("Reason for Loan", ['Cover An Unexpected Cost', 'Credit Card Refinancing', 'Home Improvement', 'Major Purchase', 'Debt Conslidation', 'Other'])
requestloanamt = st.slider("Loan Amount", min_value=1000, max_value=50000, step=100)
FICOscore = st.slider("FICO Score", min_value=0, max_value=850, step=1)
MonthlyHousingPayment = st.number_input("Monthly Housing Payment",min_value=300, max_value=49500, step=1)
Lender = st.pills("Lender", ["A", "B", "C"])
EmploymentStatus = st.pills("Employment Status", ["Full Time", "Part Time", "Unemployed"])
MonthlyGrossIncome = st.number_input("Monthly Gross Income", min_value=0, max_value=100000, step=1)

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "Reason": [reason],
    "Requested_Loan_Amount": [requestloanamt],
    "FICO_score": [FICOscore],
    "Monthly_Housing_Payment": [MonthlyHousingPayment],
    "Lender": [Lender],
    "Employment_Status": [EmploymentStatus],
    "Monthly_Gross_Income": [MonthlyGrossIncome]
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
input_data_encoded = pd.get_dummies(input_data, columns=['Reason', 'Lender', 'Employment_Status'])

# 2. Add any "missing" columns the model expects (fill with 0).
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data.
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button("Evaluate Loan"):
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]

    # Display result
    if prediction == 1:
        st.write("Your Loan is Denied.")
    else:
        st.write("Your Loan is Approved!")

st.image("https://static.vecteezy.com/system/resources/thumbnails/000/287/135/small/1__284_29.jpg")
