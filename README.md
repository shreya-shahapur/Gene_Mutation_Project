# AI-Based Gene Mutation Impact Prediction System

A machine learning project that predicts whether a genetic variant is **Pathogenic (harmful)** or **Benign (safe)** using real clinical data from ClinVar.

## Tech Stack
- Python, Jupyter Notebook
- XGBoost, Scikit-learn
- SHAP, Matplotlib, Seaborn
- Streamlit

## Dataset
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) — 830,000+ real clinical variants

## Results
- AUC-ROC: 0.789 (baseline model)
- Accuracy: 73%

## Project Structure
gene-mutation-predictor/
├── data/ ← datasets (not pushed, too large)
├── notebooks/ ← Jupyter notebooks
├── app.py ← Streamlit web app (coming soon)
└── README.md

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebook: open `notebooks/analysis.ipynb`