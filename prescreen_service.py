# prescreen_service.py
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = 'sme_prescreen_model.joblib'

def load_model(path=MODEL_PATH):
    data = joblib.load(path)
    return data['pipeline'], data['threshold']

pipeline, threshold = load_model()

def score_application(app_dict):
    """
    app_dict: dict with keys matching training features:
      annual_revenue, years_in_business, avg_monthly_profit, num_employees,
      sector, owner_age, has_existing_loan, credit_score
    Returns: dict with probability, decision, threshold
    """
    df = pd.DataFrame([app_dict])
    prob = pipeline.predict_proba(df)[:,1][0]
    decision = 'prescreen_pass' if prob >= threshold else 'prescreen_fail'
    return {'probability': float(prob), 'decision': decision, 'threshold': threshold}

if __name__ == '__main__':
    sample = {
        'annual_revenue': 3_000_000,
        'years_in_business': 5,
        'avg_monthly_profit': 60000,
        'num_employees': 10,
        'sector': 'services',
        'owner_age': 45,
        'has_existing_loan': 0,
        'credit_score': 700
    }
    print(score_application(sample))
