# Customer Churn Capstone

Predicts which bank customers are likely to churn (leave), so the
retention team can target them with offers before they go.

## Structure
```
Customer-Churn-Capstone/
├── data/
│   ├── raw/           # original CSV, never edited
│   └── processed/     # cleaned/engineered data
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/            # saved trained models
├── reports/           # EDA_Report.pdf, Model_Report.pdf (generated later)
├── app/
│   └── streamlit_app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Progress
- [x] Step 1 — Project structure, environment, dataset in place
- [ ] Step 2 — Data understanding & validation (notebook 01)
- [ ] Step 3 — EDA (notebook 02)
- [ ] Step 4 — Feature engineering (notebook 03)
- [ ] Step 5 — Model training & tuning (notebook 04, src/train.py)
- [ ] Step 6 — Evaluation & comparison (src/evaluate.py)
- [ ] Step 7 — Prediction pipeline (src/predict.py)
- [ ] Step 8 — Streamlit app (app/streamlit_app.py)
