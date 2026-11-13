
from pathlib import Path
from typing import Any

import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "best_airbnb_price_model_tuned.joblib"
SUMMARY_PATH = ROOT / "reports" / "tuning_summary.json"


# ==========================================
# 2. LOAD MODEL AND FEATURE INFORMATION
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not SUMMARY_PATH.exists():
    raise FileNotFoundError(f"Training summary not found: {SUMMARY_PATH}")

model = joblib.load(MODEL_PATH)

with open(SUMMARY_PATH, "r", encoding="utf-8") as file:
    summary = json.load(file)

FEATURES = summary["features_used"]


# ==========================================
# 3. INITIALIZE FASTAPI
# ==========================================

app = FastAPI(
    title="Airbnb Price Prediction API",
    description=(
        "Machine learning API for predicting "
        "Airbnb listing prices in India."
    ),
    version="1.0.0"
)


# ==========================================
# 4. REQUEST SCHEMA
# ==========================================

class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Airbnb listing features matching the training columns."
    )


# ==========================================
# 5. ROOT ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Welcome to Airbnb Price Prediction API",
        "status": "running",
        "model": summary["best_model"],
        "version": "1.0.0"
    }


# ==========================================
# 6. HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "features_loaded": len(FEATURES)
    }


# ==========================================
# 7. MODEL INFORMATION
# ==========================================

@app.get("/model-info")
def model_info():

    return {
        "model_name": summary["best_model"],
        "features_count": len(FEATURES),
        "features": FEATURES,
        "test_metrics": summary.get(
            "random_holdout_test_metrics", {}
        ),
        "cross_validation": summary.get(
            "geographic_group_cv", {}
        )
    }


# ==========================================
# 8. FEATURE INFORMATION
# ==========================================

@app.get("/features")
def get_features():

    return {
        "required_features": FEATURES,
        "total_features": len(FEATURES),
        "instructions": (
            "Send all required features in the "
            "prediction request."
        )
    }


# ==========================================
# 9. PRICE PREDICTION
# ==========================================

@app.post("/predict")
def predict_price(request: PredictionRequest):

    try:

        input_features = request.features

        # Check missing features
        missing_features = [
            feature
            for feature in FEATURES
            if feature not in input_features
        ]

        if missing_features:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Missing required features",
                    "missing_features": missing_features
                }
            )

        # Keep only expected features in training order
        input_data = {
            feature: input_features[feature]
            for feature in FEATURES
        }

        # Convert input into DataFrame
        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURES
        )

        # Generate prediction
        prediction = model.predict(input_df)[0]

        prediction = float(prediction)

        if not pd.notna(prediction):
            raise ValueError("Model returned an invalid prediction")

        return {
            "success": True,
            "predicted_price": round(prediction, 2),
            "model_used": summary["best_model"],
            "price_unit": "Source dataset units"
        }

    except HTTPException:
        raise

    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(
            status_code=422,
            detail=str(error)
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ==========================================
# 10. SERVER INFORMATION
# ==========================================

@app.get("/about")
def about():

    return {
        "project": "AirBnBPrice",
        "purpose": "Airbnb price prediction in India",
        "algorithm": summary["best_model"],
        "deployment": "FastAPI",
        "model_path": str(MODEL_PATH)
    }
