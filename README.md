# Customer Churn Pipeline

## Problem Statement
A financial services company (Personal Loans, Credit Cards, Savings Accounts,
Insurance, Investment Services) wants to identify customers who are likely to
churn (stop using services / not renew products), so the retention team can
proactively target them with offers and campaigns before they leave.

This project builds an end-to-end, production-style machine learning pipeline
that predicts the **probability of churn** for each customer.

## Project Status
🚧 Phase 1: Project setup and planning — in progress.

## Project Structure
```
Customer-Churn-Pipeline/
├── data/
│   ├── raw/            # Immutable original data (never edited directly)
│   └── processed/       # Cleaned / feature-engineered data
├── notebooks/           # Exploratory analysis (not production code)
├── src/                 # Reusable, production-quality Python modules
├── models/              # Serialized trained models (.pkl / .joblib)
├── reports/             # EDA reports, evaluation metrics, figures
├── app/                 # Streamlit / FastAPI deployment code
├── tests/                # Unit tests for src/ modules
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Roadmap
- [x] Phase 1 — Planning, environment, repo setup
- [ ] Phase 2 — Data collection & validation
- [ ] Phase 3 — Exploratory Data Analysis
- [ ] Phase 4 — Data cleaning & feature engineering
- [ ] Phase 5 — Feature selection & preprocessing pipeline
- [ ] Phase 6 — Model training & hyperparameter tuning
- [ ] Phase 7 — Model evaluation & comparison
- [ ] Phase 8 — Serialization & prediction pipeline
- [ ] Phase 9 — Deployment (Streamlit + FastAPI)
