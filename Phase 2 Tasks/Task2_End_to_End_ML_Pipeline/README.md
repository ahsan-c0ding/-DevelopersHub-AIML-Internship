# Telco Customer Churn — End-to-End sklearn Pipeline

## Objective

Build a reusable, production-ready machine learning pipeline that predicts customer churn
for a telecom provider, using scikit-learn's `Pipeline` / `ColumnTransformer` API so that
preprocessing (scaling, encoding) and modeling are bundled into a single exportable object , 
no separate scaler/encoder files to keep in sync at inference time.

## Dataset

[Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn):
7,043 customers, 20 features (demographics, subscribed services, account/billing info), and
a binary `Churn` label. ~26.5% of customers in the dataset churned.

## Methodology / Approach

1. **Load & clean**:  fix `TotalCharges` (loaded as text due to 11 blank entries for
   brand-new customers with zero tenure), drop the `customerID` identifier, map `Churn` to
   0/1.
2. **EDA**:  check class balance, and look at churn rate by contract type and tenure to
   sanity-check the data before modeling.
3. **Preprocessing pipeline**: a `ColumnTransformer` applying `StandardScaler` to numeric
   features (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`) and `OneHotEncoder`
   to the remaining categorical features.
4. **Models**: Logistic Regression and Random Forest, each wrapped with the preprocessor in
   a single `Pipeline`.
5. **Hyperparameter tuning**: `GridSearchCV` (5-fold, scored on ROC-AUC since the classes
   are moderately imbalanced) over regularization strength / class weighting for Logistic
   Regression, and tree count / depth / split size / class weighting for Random Forest.
6. **Evaluation**: accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC
   curves on a held-out 20% test set.
7. **Export**: the best pipeline is serialized with `joblib` as a single file that accepts
   raw customer rows and returns predictions directly.

## Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.805 | 0.655 | 0.559 | 0.603 | **0.841** |
| Random Forest | 0.774 | 0.557 | 0.717 | 0.627 | 0.838 |

- **Logistic Regression** was selected as the exported model, best ROC-AUC and accuracy,
  and it's simpler and faster to serve than the Random Forest.
- **Random Forest** trades precision for recall (catches more actual churners, at the cost
  of more false positives), worth reconsidering if the business would rather over-flag
  than under-flag risky customers.
- **Biggest churn drivers:** contract type (month-to-month churns far more than 1–2 year
  contracts), tenure (new customers are the highest risk), and monthly charges. Add-on
  services like tech support and online security correlate with lower churn.

See `churn_pipeline_notebook.ipynb` for the full analysis, charts, and metrics.

## Repository Structure

```
.
├── churn_pipeline_notebook.ipynb   # full walkthrough: EDA, pipeline, tuning, evaluation
├── churn_pipeline.joblib           # exported, ready-to-use trained pipeline
├── predict_example.py              # minimal script showing how to reuse the pipeline
├── Telco-Customer-Churn.csv        # dataset (needed to re-run the notebook)
├── requirements.txt
└── README.md
```

## Usage

```bash
pip install -r requirements.txt
python predict_example.py
```

```python
import joblib
import pandas as pd

pipeline = joblib.load("churn_pipeline.joblib")
pipeline.predict(new_customer_dataframe)          # 0/1 churn prediction
pipeline.predict_proba(new_customer_dataframe)     # churn probability
```

`new_customer_dataframe` just needs the same raw columns as the original CSV (minus
`customerID` and `Churn`) the pipeline handles scaling and encoding internally.

## Next Steps

- SHAP values for per-customer explanations of *why* a prediction was made
- Cost-sensitive threshold tuning based on the real $ cost of a missed churner vs. a
  retention outreach call
- Try gradient-boosted trees (XGBoost/LightGBM) alongside LR and RF
