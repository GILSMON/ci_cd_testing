from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = None


class HousingFeatures(BaseModel):
    MedInc: float        # Median income in block group
    HouseAge: float      # Median house age in block group
    AveRooms: float      # Average number of rooms per household
    AveBedrms: float     # Average number of bedrooms per household
    Population: float    # Block group population
    AveOccup: float      # Average number of household members
    Latitude: float
    Longitude: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load("model.pkl")
    print("Model loaded successfully")
    yield


app = FastAPI(title="Housing Price Predictor", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(features: HousingFeatures):
    data = [[
        features.MedInc,
        features.HouseAge,
        features.AveRooms,
        features.AveBedrms,
        features.Population,
        features.AveOccup,
        features.Latitude,
        features.Longitude,
    ]]
    prediction = model.predict(data)[0]
    return {"prediction": round(prediction, 4)}
