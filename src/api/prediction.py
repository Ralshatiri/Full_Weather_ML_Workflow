"""
FastAPI endpoints for temperature forecasting.

The API validates forecast requests, checks Redis for cached results,
and publishes prediction jobs to RabbitMQ.

"""

import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.services.rabbitmq_publisher import publish_prediction_job
from src.services.redis_cache import (
    connect_redis,
    get_cached_prediction,
    get_job_status,
    make_cache_key,
    save_job_status,
)


load_dotenv()

connect_redis()


app = FastAPI(
    title="Temperature Forecast API",
    version="1.0.0",
)


class ForecastInput(BaseModel):
    """
    Define one temperature forecast request.

    The user supplies a supported city and the number of future days to
    forecast.
    """

    city: str

    forecast_days: int

    @field_validator("city")
    @classmethod
    def clean_city_name(cls, city: str) -> str:
        """
        Remove surrounding whitespace and reject an empty city name.

        Parameters
        ----------
        city : str
            City name received from the API request.

        Returns
        -------
        str
            Cleaned city name.

        Raises
        ------
        ValueError
            If the city name contains only whitespace.
        """
        cleaned_city = city.strip()

        if not cleaned_city:
            raise ValueError(
                "City must not be empty."
            )

        return cleaned_city


@app.get("/health")
def health() -> dict:
    """
    Return current API health status.
    """
    return {
        "status": "running"
    }


@app.post("/predict/single")
def predict_single(data: ForecastInput) -> dict:
    """
    The endpoint checks whether the same request already has a cached result.
    If not, it creates a job identifier, records the pending status in Redis,
    and publishes the request to RabbitMQ for worker processing.
    """
    try:
        input_dict = data.model_dump()

        cache_key = make_cache_key(
            "forecast:recursive:single",
            input_dict,
        )

        cached_response = get_cached_prediction(
            cache_key
        )

        if cached_response:
            return {
                "status": "completed",
                "source": "redis_cache",
                "result": cached_response,
            }

        job_id = str(uuid.uuid4())

        job_message = {
            "job_id": job_id,
            "job_type": "single_prediction",
            "cache_key": cache_key,
            "input": input_dict,
        }

        save_job_status(
            job_id,
            {
                "status": "pending",
                "job_id": job_id,
                "job_type": "single_prediction",
                "message": (
                    "Recursive forecast job is waiting "
                    "to be processed."
                ),
            },
        )

        publish_prediction_job(job_message)

        return {
            "status": "queued",
            "job_id": job_id,
            "message": (
                "Forecast job queued. Check the result using "
                f"/predict/result/{job_id}"
            ),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Forecast queueing failed: "
                f"{error}"
            ),
        ) from error


@app.post("/predict/batch")
def predict_batch(
    data: list[ForecastInput],
) -> dict:
    """
    Queue forecasts for multiple cities in one job.

    Each list item can request a different city and forecast length. The
    complete batch is processed asynchronously by the prediction worker.
    """
    try:
        if not data:
            raise HTTPException(
                status_code=400,
                detail=(
                    "At least one forecast request "
                    "must be provided."
                ),
            )

        input_list = [
            row.model_dump()
            for row in data
        ]

        cache_key = make_cache_key(
            "forecast:recursive:batch",
            {"items": input_list},
        )

        cached_response = get_cached_prediction(
            cache_key
        )

        if cached_response:
            return {
                "status": "completed",
                "source": "redis_cache",
                "result": cached_response,
            }

        job_id = str(uuid.uuid4())

        job_message = {
            "job_id": job_id,
            "job_type": "batch_predict",
            "cache_key": cache_key,
            "input": input_list,
        }

        save_job_status(
            job_id,
            {
                "status": "pending",
                "job_id": job_id,
                "job_type": "batch_predict",
                "message": (
                    "Batch recursive forecast job is "
                    "waiting to be processed."
                ),
            },
        )

        publish_prediction_job(job_message)

        return {
            "status": "queued",
            "job_id": job_id,
            "message": (
                "Batch forecast job queued. Check the result using "
                f"/predict/result/{job_id}"
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Batch forecast queueing failed: "
                f"{error}"
            ),
        ) from error


@app.get("/predict/result/{job_id}")
def get_prediction_result(job_id: str) -> dict:
    """
    Retrieve the current status or completed result of a forecast job.
    """
    try:
        job_status = get_job_status(job_id)

        if not job_status:
            raise HTTPException(
                status_code=404,
                detail="Job not found or expired.",
            )

        return job_status

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to get job result: {error}"
            ),
        ) from error