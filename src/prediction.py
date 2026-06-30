from fastapi import FastAPI
import os
import joblib
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

DB_CONN = "postgresql://{}:{}@{}:{}/{}".format(
    os.getenv("POSTGRES_USER"),
    os.getenv("POSTGRES_PASSWORD"),
    os.getenv("DB_HOST"),
    os.getenv("DB_PORT"),
    os.getenv("POSTGRES_DB")
)

engine = create_engine(DB_CONN)

path = "/app/model/xgboost_model.joblib"

model = joblib.load(path)

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

@app.get("/")
def api():
    return {"message":"The API is running"}

@app.post("/predict/single")
def predict_single(data:weather_input):

    input_df = pd.DataFrame([data.model_dump()])

    prediction = float(model.predict(input_df)[0])

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

@app.post("/predict/batch")
def predict_batch(data: list[weather_input]):

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


