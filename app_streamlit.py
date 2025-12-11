# app_streamlit.py
import streamlit as st
from prescreen_service import score_application
import pandas as pd

st.title("SME Loan Prescreen Demo")

with st.form("application"):
    st.subheader("Applicant details")
    annual_revenue = st.number_input("Annual revenue", value=2000000.0, step=100000.0)
    years_in_business = st.number_input("Years in business", value=5, step=1)
    avg_monthly_profit = st.number_input("Avg monthly profit", value=50000.0, step=1000.0)
    num_employees = st.number_input("Number of employees", value=10, step=1)
    sector = st.selectbox("Sector", ["manufacturing","services","retail","agri","tech"])
    owner_age = st.number_input("Owner age", value=40, step=1)
    has_existing_loan = st.selectbox("Has existing loan?", [0,1], format_func=lambda x: "No" if x==0 else "Yes")
    credit_score = st.number_input("Credit score", value=650, step=1)

    submitted = st.form_submit_button("Prescreen")

if submitted:
    app = {
        'annual_revenue': annual_revenue,
        'years_in_business': years_in_business,
        'avg_monthly_profit': avg_monthly_profit,
        'num_employees': num_employees,
        'sector': sector,
        'owner_age': owner_age,
        'has_existing_loan': has_existing_loan,
        'credit_score': credit_score
    }
    res = score_application(app)
    st.write("Probability (model confidence):", round(res['probability'], 4))
    st.write("Decision:", res['decision'])
    st.info(f"Threshold used: {res['threshold']}")

    # Simple explanation: feature contributions using SHAP would be better (see note)
    # st.write("Tip: add SHAP explanations for per-applicant explainability.")
    st.write("~Suresh Das AEC.22")
