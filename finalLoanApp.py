
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("/content/final_loan_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
#st.title("Your Loan Approval")
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #F0FFFF;'><b>Your Loan Approval</b></h1>",
    unsafe_allow_html=True
)

st.header("Enter Loan Details")

reason = st.selectbox("Reason for Loan (REASON)", ['Cover An Unexpected Cost', 'Credit Card Refinancing', 'Home Improvement',
                                                   'Major Purchase', 'Debt Conslidation', 'Other'])
requestloanamt = st.slider("Loan Amount (LOAN)", min_value=1000, max_value=500000, step=1000)
FICOscore = st.slider("FICO Score", min_value=300, max_value=850, step=1)
MonthlyHousingPayment = st.number_input("Monthly Housing Payment",min_value=300, max_value=49500, step=1)
Lender = st.selectbox("Lender", ["A", "B", "C"])
EverBankruptorForeclose = st.checkbox("Ever Bankrupt or Foreclosed")
EmploymentStatus = st.selectbox("Employment Status", ["Full Time", "Part Time", "Unemployed"])
EmploymentSector = st.selectbox("Employment Sector", ['Consumer Discretionary', 'Information Technology', 'Energy',
                                                      'Consumer Staples', 'Communication Services', 'Materials' 'Utilities',
                                                      'Real Estate', 'Health Care', 'Industrials', 'Financials', "Unknown"])
MonthlyGrossIncome = st.number_input("Monthly Gross Income", min_value=0, max_value=100000, step=1)

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "Reason": [reason],
    "Requested_Loan_Amount": [requestloanamt],
    "FICO_score": [FICOscore],
    "Monthly_Housing_Payment": [MonthlyHousingPayment],
    "Lender": [Lender],
    "Ever_Bankrupt_or_Foreclose": [EverBankruptorForeclose],
    "Employment_Status": [EmploymentStatus],
    "Employment_Sector": [EmploymentSector],
    "Monthly_Gross_Income": [MonthlyGrossIncome]
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
input_data_encoded = pd.get_dummies(input_data, columns=['Reason', 'Lender', 'Employment_Status', 'Employment_Sector'])

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
        st.write("The prediction is: **Bad Loan** 🚫")
    else:
        st.write("The prediction is: **Good Loan** 💲")



        """
What happens if the user enters a value not in the training data?

Example: User enters REASON = 'Vacation', but the model only knows 'DebtCon' and 'HomeImp'.

1. pd.get_dummies creates a new column: REASON_Vacation = 1.
2. The code then adds the *known* columns: REASON_DebtCon = 0 and REASON_HomeImp = 0.
3. The final filtering step *drops* the unknown REASON_Vacation column because it's not in the
   model's expected feature list.

Result: The model receives REASON_DebtCon = 0 and REASON_HomeImp = 0, which correctly
treats the unknown 'Vacation' input as "none of the known categories" (i.e., "Other").
"""
