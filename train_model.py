# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix
import joblib

DATA_PATH = 'sme_synthetic.csv'  # replace with your real CSV

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # basic cleanup: drop rows with missing label (if any)
    df = df.dropna(subset=['label'])
    return df

def build_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    ])
    pre = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])
    return pre

def train_and_evaluate(df):
    target = 'label'
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = ['annual_revenue','years_in_business','avg_monthly_profit','num_employees','owner_age','credit_score','has_existing_loan']
    cat_cols = ['sector']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    pre = build_preprocessor(num_cols, cat_cols)

    # Candidate models
    models = {
        'logreg': Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=1000))]),
        'rf': Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))])
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, probs)
        preds = (probs >= 0.5).astype(int)
        prec = precision_score(y_val, preds)
        rec = recall_score(y_val, preds)
        acc = accuracy_score(y_val, preds)
        results[name] = {'pipeline': pipe, 'auc': auc, 'precision': prec, 'recall': rec, 'accuracy': acc}
        print(f"{name}: AUC={auc:.4f}, precision={prec:.4f}, recall={rec:.4f}, acc={acc:.4f}")

    # pick best by AUC
    best_name = max(results.keys(), key=lambda k: results[k]['auc'])
    best = results[best_name]
    print("Selected model:", best_name)

    # Evaluate on test
    test_probs = best['pipeline'].predict_proba(X_test)[:,1]
    test_auc = roc_auc_score(y_test, test_probs)
    test_preds = (test_probs >= 0.5).astype(int)
    print("Test AUC:", test_auc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))
    print("Test precision/recall:", precision_score(y_test, test_preds), recall_score(y_test, test_preds))

    # Save model and a chosen threshold (we'll store default 0.5; business can change)
    joblib.dump({'pipeline': best['pipeline'], 'threshold': 0.5}, 'sme_prescreen_model.joblib')
    print("Model saved to sme_prescreen_model.joblib")

    return best, results

if __name__ == '__main__':
    df = load_data()
    best, results = train_and_evaluate(df)
