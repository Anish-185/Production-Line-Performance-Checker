from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")

class InputData(BaseModel):
    Type: int
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: float

@app.get("/")
def home():
    return {"message": "Production Line Performance Checker"}

@app.post("/predict")
def predict(data: InputData):

    features = np.array([[
        data.Type,
        data.Air_temperature,
        data.Process_temperature,
        data.Rotational_speed,
        data.Torque,
        data.Tool_wear
    ]])

    features = scaler.transform(features)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "machine_failure": int(prediction),
        "failure_probability": round(float(probability), 4)
    }