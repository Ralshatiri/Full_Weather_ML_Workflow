from fastapi import FastAPI
from fastapi import HTTPException
import os
import joblib
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine
from src.config import DB_CONN, MODEL_PATH

load_dotenv()


engine = create_engine(DB_CONN)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(
        f"Failed to load model: {str(e)}"
    )

# create the fastapi
app = FastAPI()

class weather_input(BaseModel):
    city: str
    relative_humidity_2m: float
    precipitation: float
    weather_code: int
    surface_pressure: float
    cloud_cover: float
    wind_speed_10m: float
    wind_direction_10m: float
    wind_gusts_10m: float
    month: int
    day: int
    hour: int
    temperature_lag_24: float
    temperature_rolling_mean_24: float
    temperature_rolling_std_24: float
    temperature_lag_720: float

@app.get("/")
def home():
    return {
        "message":"The API is running"
        }

@app.get("/health")
def health():
    return {
        "status":"running"
    }


@app.post("/predict/single")
def predict_single(data:weather_input):

    try:

        input_df = pd.DataFrame([data.model_dump()])

        prediction = round(float(model.predict(input_df)[0]),2)

        result_df = input_df.copy()
        result_df["predicted_temperature"] = prediction

        result_df.to_sql(
            name="predictions",
            con=engine,
            if_exists="append",
            index=False
        )

        return {
            "predicted_temperature":prediction,
            "message":"Single prediction saved to database"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Single prediction failed: {str(e)} "

        )

@app.post("/predict/batch")
def predict_batch(data: list[weather_input]):

    try:

        input_df = pd.DataFrame([row.model_dump() for row in data])

        predictions = model.predict(input_df)

        resultes_df = input_df.copy()
        resultes_df["predicted_temperature"] = predictions

        resultes_df.to_sql(
            name="predictions",
            con=engine,
            if_exists='append',
            index=False
        )

        return {
            "Number of predictions: ":len(predictions),
            "Predictions": [round(float(pred),2) for pred in predictions],
            "message":"Batch prediction saved to database"

        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )

