# ==============================
# FASTAPI MODEL SERVER
# ==============================

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib


# ------------------------------
# 1. LOAD MODEL
# ------------------------------
model = joblib.load("loan_random_forest_pipeline.pkl")


# ------------------------------
# 2. FASTAPI APP
# ------------------------------
app = FastAPI(title="Loan Approval Prediction API")


# ------------------------------
# 3. INPUT SCHEMA
# ------------------------------
class LoanApplication(BaseModel):
    no_of_dependents: int
    education: str          # "Graduate" / "Not Graduate"
    self_employed: str      # "Yes" / "No"
    income_annum: int
    loan_amount: int
    loan_term: int
    cibil_score: int


# ------------------------------
# 4. HEALTH CHECK
# ------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ------------------------------
# 5. PREDICTION ENDPOINT
# ------------------------------
@app.post("/predict")
def predict(application: LoanApplication):

    # Convert input to DataFrame (model expects DataFrame)
    input_df = pd.DataFrame([application.dict()])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "loan_status": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": round(float(probability), 4)
    }
