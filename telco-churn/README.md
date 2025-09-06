# Telco Customer Churn – End-to-End ML + Streamlit

**What this does**  
Predicts customer churn using the classic Telco dataset. Includes:
- Reproducible training (`train_churn.py`)
- Web app for single/batch predictions (`app.py`)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate # Windows (.venv\Scripts\activate)
pip install -r requirements.txt
# Put dataset at: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
python train_churn.py            # trains and saves artifacts/
streamlit run app.py             # runs the UI
