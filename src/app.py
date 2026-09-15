import joblib
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Define the incoming Data Structure (Data Validation)
class CustomerData(BaseModel):
    first_checkout_amount: float
    pages_viewed_week1: int
    discount_used_week1: int
    category_first_item: str

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        ml_models["model"] = joblib.load("models/ltv_model.pkl")
        ml_models["features"] = joblib.load("models/model_features.pkl")
        print("ML Model and features successfully loaded into memory!")
    except FileNotFoundError:
        print("Error: Model files not found! Please run src/train.py first.")
    yield
    ml_models.clear()    

# 2. Initialize FastAPI app
app = FastAPI(title="Customer LTV Prediction Service", version="1.0", lifespan=lifespan)

@app.post("/predict")
def predict_ltv(data: CustomerData):
    """API Endpoint to predict 12-month LTV based on the 1st week of user logs."""
    if"model" not in ml_models or "features" not in ml_models:
        raise HTTPException(status_code=500, detail="ML model is not loaded on the server.")
    
# 1. Convert incoming JSON request to a Pandas DataFrame row    
    raw_input = pd.DataFrame([data.model_dump()])
# 2. Apply EXACTLY the same Feature Engineering from training
    raw_input['feature_engagement_index'] = raw_input['pages_viewed_week1'] * (raw_input['first_checkout_amount'] / 100.0)
    raw_input['feature_ios_premium_wallet'] = ((raw_input['device_type'] == 'iOS') & (raw_input['discount_used_week1'] == 0)).astype(int)
    raw_input['feature_intent_score'] = (raw_input['first_checkout_amount'] * raw_input['pages_viewed_week1']) / (raw_input['discount_used_week1'] + 1)

# 3. Match the One-Hot Encoded columns structure used during model.fit()    
    encoded_input = pd.get_dummies(raw_input)
# Fill missing encoded categorical columns with 0s and align order
    processed_input = pd.DataFrame(0, index=[0], columns=ml_models["features"])
    for col in encoded_input.columns:
        if col in processed_input.columns:
            processed_input[col] = encoded_input[col].values

# 4. Make prediction and inverse transform from log scale back to original scale
    predicted_log = ml_models["model"].predict(processed_input)
    final_ltv_prediction = np.expm1(predicted_log)[0]

# 5. Return JSON response 
    return {
        "status": "success",
        "predicted_12m_ltv": round(float(final_ltv_prediction), 2)
        }