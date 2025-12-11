# generate_synthetic_data.py
import numpy as np
import pandas as pd

def generate(n=2000, seed=42):
    np.random.seed(seed)
    # Simple SME features
    annual_revenue = np.random.lognormal(mean=12, sigma=1.0, size=n)  # in INR units or any
    years_in_business = np.random.poisson(lam=6, size=n)
    avg_monthly_profit = np.random.normal(loc=50000, scale=30000, size=n)
    num_employees = np.random.poisson(lam=12, size=n)
    sector = np.random.choice(['manufacturing','services','retail','agri','tech'], size=n, p=[0.2,0.3,0.25,0.15,0.1])
    owner_age = np.random.normal(40, 10, size=n).astype(int)
    has_existing_loan = np.random.choice([0,1], size=n, p=[0.7,0.3])
    credit_score = np.clip(np.random.normal(650, 70, size=n).astype(int), 300, 900)

    # Create a risk score (synthetic) -> label: 1=pass prescreen, 0=reject
    risk = (
        (annual_revenue / annual_revenue.max()) * 0.3
        + (np.clip(years_in_business,0,20)/20)*0.2
        + (np.clip(avg_monthly_profit, -1e6, 1e6) / (avg_monthly_profit.max())) * 0.2
        + (credit_score/900)*0.2
        - has_existing_loan*0.3
    )
    # add sector effect
    sector_map = {'manufacturing':0.02, 'services':0.0, 'retail':-0.02, 'agri':-0.05, 'tech':0.05}
    risk += np.array([sector_map[s] for s in sector])

    probs = 1 / (1 + np.exp(-5*(risk - np.median(risk))))  # squash to (0,1)
    label = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        'annual_revenue': annual_revenue,
        'years_in_business': years_in_business,
        'avg_monthly_profit': avg_monthly_profit,
        'num_employees': num_employees,
        'sector': sector,
        'owner_age': owner_age,
        'has_existing_loan': has_existing_loan,
        'credit_score': credit_score,
        'label': label
    })
    return df

if __name__ == '__main__':
    df = generate(3000)
    df.to_csv('sme_synthetic.csv', index=False)
    print("Saved sme_synthetic.csv with", len(df), "rows")
