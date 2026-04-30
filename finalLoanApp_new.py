
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

reason = st.selectbox("Reason", [
    "cover_an_unexpected_cost",
    "credit_card_refinancing",
    "home_improvement",
    "major_purchase",
    "debt_conslidation",
    "other"
])
requestloanamt = st.slider("Loan Amount", min_value=1000, max_value=50000, step=100)
FICOscore = st.slider("FICO Score", min_value=0, max_value=850, step=1)
EverBankruptOrForeclose = st.checkbox("Ever Bankrupt or Foreclosed")
MonthlyHousingPayment = st.number_input("Monthly Housing Payment",min_value=300, max_value=49500, step=1)
lender = st.selectbox("Lender", ["A", "B", "C"])
employment_status = st.selectbox("Employment Status", [
    "full_time",
    "part_time"
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
MonthlyGrossIncome = st.number_input("Monthly Gross Income", min_value=0, max_value=100000, step=1)

# Categorize FICO score into groups
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

# Create the input data as a DataFrame
input_data = pd.DataFrame([{
    "Reason": reason,
    "Granted_Loan_Amount": requestloanamt,   # ⚠️ must match training column name
    "Requested_Loan_Amount": requestloanamt,
    "FICO_score": FICOscore,
    "Fico_Score_group": fico_group(FICOscore),
    "Employment_Status": employment_status,
    "Employment_Sector": employment_sector,
    "Monthly_Gross_Income": MonthlyGrossIncome,
    "Monthly_Housing_Payment": MonthlyHousingPayment,
    "Ever_Bankrupt_or_Foreclose": int(EverBankruptOrForeclose),
    "Lender": lender
}])

input_data['FICO_score_group'] = input_data['FICO_score'].apply(fico_group)
input_data["Reason"] = input_data["Reason"].str.strip().str.lower().str.replace(" ", "_")
input_data["Lender"] = input_data["Lender"].str.strip().str.lower()
input_data["Employment_Status"] = input_data["Employment_Status"].str.strip().str.lower().str.replace(" ", "_")

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
# Add 'FICO_score_group' to the columns to be one-hot encoded
input_data_encoded = pd.get_dummies(input_data, drop_first=True)

input_data_encoded = input_data_encoded.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

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

    st.write(input_data_encoded)
    st.write("Prediction:", prediction)
    st.write("Probabilities:", model.predict_proba(input_data_encoded))
    st.write(input_data_encoded)
    leaf_id = model.apply(input_data_encoded)[0]
    st.write("Leaf node:", leaf_id)
    st.write("Non-zero features:")
    st.write(input_data_encoded.loc[:, (input_data_encoded != 0).any(axis=0)])

    importance = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    st.write(importance.sort_values(ascending=False).head(10))

    # Display result
    if prediction == 1:
        st.write("Your Loan is Approved!")
    else:
        st.write("Your Loan is Denied.")

    st.write(input_data_encoded.T)
    st.write("Leaf:", model.apply(input_data_encoded)[0])

    st.write("Debug:")
    st.write(model.feature_names_in_)
    st.write(input_data_encoded.columns)
    st.write("Non-zero features:")
    st.write(input_data_encoded.loc[:, input_data_encoded.sum() > 0])

st.image("https://static.vecteezy.com/system/resources/thumbnails/000/287/135/small/1__284_29.jpg")
