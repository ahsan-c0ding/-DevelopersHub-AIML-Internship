"""
Reuse example for churn_pipeline.joblib.

The whole point of exporting a Pipeline (not just a bare model) is that this script
doesn't need to know anything about StandardScaler or OneHotEncoder — raw customer
rows go in, predictions come out. Run: python predict_example.py
"""
import pandas as pd
import joblib

pipeline = joblib.load("churn_pipeline.joblib")

# a couple of made-up customers, shaped exactly like the raw CSV columns minus customerID/Churn
new_customers = pd.DataFrame([
    {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 190.0,
    },
    {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
    },
])

predictions = pipeline.predict(new_customers)
probabilities = pipeline.predict_proba(new_customers)[:, 1]

for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
    label = "WILL CHURN" if pred == 1 else "will stay"
    print(f"Customer {i}: {label}  (churn probability: {proba:.2%})")
