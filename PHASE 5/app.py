import pandas as pd
import joblib
import logging
import hashlib
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict

# 1. Setup Logging
logging.basicConfig(filename='api_logs.log', level=logging.INFO)

app = FastAPI(title="Hospital Prediction API")

# 2. Load Models
risk_model = joblib.load('risk_model.joblib')
claim_model = joblib.load('claim_model.joblib')

# 3. Schema Validation (Matches Phase 3 Feature Schema)
class PredictionRequest(BaseModel):
    features: Dict[str, float]

@app.get("/health")
def health_check():
    return {"status": "online", "timestamp": datetime.now()}

@app.post("/predict")
def predict(request: PredictionRequest, model_type: str):
    try:
        # Create hash for audit log
        input_data = pd.DataFrame([request.features])
        feature_hash = hashlib.md5(input_data.to_string().encode()).hexdigest()
        
        # Select model
        model = risk_model if model_type == "risk" else claim_model
        prediction = model.predict(input_data)[0]
        
        # Log entry
        logging.info(f"Model: {model_type} | Hash: {feature_hash} | Result: {prediction} | Time: {datetime.now()}")
        
        return {"model": model_type, "prediction": str(prediction), "audit_hash": feature_hash}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))