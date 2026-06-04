import joblib
import pandas as pd
from pathlib import Path


BASE_DIR  =  Path(__file__).resolve().parent.parent
MODEL_PATH  =  BASE_DIR / "models" / "churn_model.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def predict_churn(input_data: dict):
    model  =  load_model()

    input_df  =  pd.DataFrame([input_data]) 

    prediction = model.predict(input_df)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_df)[0][1]

    else:
        probability  = None

    result =  {
                "prediction": int(prediction),
                "churn_label": "Yes" if prediction == 1 else "No",
                "churn_probability": round(float(probability), 4) if probability is not None else None
    }

    return result


if __name__ == "__main__":

    sample_customer = {
        "Gender": "Female",
        "Senior Citizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure Months": 12,
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "Yes",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Monthly Charges": 85.5,
        "Total Charges": 1026.0,
        "CLTV": 3500
    }


    output   =   predict_churn(sample_customer)
    print("\nPrediction Result:")
    print(output)