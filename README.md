# Wine Quality Assesment

Portfolio-grade Streamlit app for binary wine quality prediction.

## Created by
Powell Ndlovu

## Features
- Random Forest and AdaBoost compared side by side
- Top-level metric cards
- Random Forest feature importance chart
- SHAP local explanation panel
- Batch CSV prediction with schema and numeric validation
- Downloadable CSV template and scored outputs

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Required input columns
fixed_acidity, volatile_acidity, citric_acidity, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol
