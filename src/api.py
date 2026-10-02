from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from pathlib import Path

app = FastAPI(title="Business Growth Prediction API")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "gradient_boosting_model.pkl"

pipeline = joblib.load(MODEL_PATH)


class BusinessInput(BaseModel):
    Category: str
    City: str
    Latitude: float = 29.9457
    Longitude: float = 78.1642
    Year_Opened: int = 2021
    Age: int = 3
    Capacity: int
    Observation_Year: int = 2024
    Nearby_Population_2km: float = 15000
    Competitors_500m: int = 2
    Competitors_1km: int = 5
    Competitors_2km: int = 12
    Closure_Rate_2km: float = 0.04
    Demand_Index: float
    New_Business_Rate: float = 0.12
    Revenue_2022_INR: float

@app.post("/predict")
def predict_growth(data: BusinessInput):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = pipeline.predict(df)[0]
        return {"predicted_growth_12m": round(float(prediction), 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
