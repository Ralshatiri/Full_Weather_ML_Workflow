from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid

from src.services.redis_cache import (
    connect_redis,
    make_cache_key,
    get_cached_prediction,
    save_job_status,
    get_job_status,
)

from src.services.rabbitmq_publisher import publish_prediction_job

load_dotenv()




connect_redis()


# ---------- FastAPI app ----------
app = FastAPI()


class WeatherInput(BaseModel):
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
        "message": "The API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "running"
    }


@app.post("/predict/single")
def predict_single(data: WeatherInput):
    try:
        input_dict = data.model_dump()

        cache_key = make_cache_key("prediction:single", input_dict)

        cached_response = get_cached_prediction(cache_key)

        if cached_response:
            return {
                "status": "completed",
                "source": "redis_cache",
                "result": cached_response
            }

        job_id = str(uuid.uuid4())

        job_message = {
            "job_id": job_id,
            "job_type": "single_prediction",
            "cache_key": cache_key,
            "input": input_dict
        }

        save_job_status(job_id, {
            "status": "pending",
            "job_id": job_id,
            "job_type": "single_prediction",
            "message": "Prediction job is waiting to be processed."
        })

        publish_prediction_job(job_message)

        return {
            "status": "queued",
            "job_id": job_id,
            "message": "Single prediction job queued. Check result using /predict/result/{job_id}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Single prediction queueing failed: {str(e)}"
        )


@app.post("/predict/batch")
def predict_batch(data: list[WeatherInput]):
    try:
        input_list = [row.model_dump() for row in data]

        cache_key = make_cache_key("prediction:batch", {"items": input_list})

        cached_response = get_cached_prediction(cache_key)

        if cached_response:
            return {
                "status": "completed",
                "source": "redis_cache",
                "result": cached_response
            }

        job_id = str(uuid.uuid4())

        job_message = {
            "job_id": job_id,
            "job_type": "batch_prediction",
            "cache_key": cache_key,
            "input": input_list
        }

        save_job_status(job_id, {
            "status": "pending",
            "job_id": job_id,
            "job_type": "batch_prediction",
            "message": "Batch prediction job is waiting to be processed."
        })

        publish_prediction_job(job_message)

        return {
            "status": "queued",
            "job_id": job_id,
            "message": "Batch prediction job queued. Check result using /predict/result/{job_id}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction queueing failed: {str(e)}"
        )


@app.get("/predict/result/{job_id}")
def get_prediction_result(job_id: str):
    try:
        job_status = get_job_status(job_id)

        if not job_status:
            raise HTTPException(
                status_code=404,
                detail="Job not found or expired."
            )

        return job_status

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get job result: {str(e)}"
        )