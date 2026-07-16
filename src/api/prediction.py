"""
FastAPI endpoints for temperature forecasting.

The API validates forecast requests, checks Redis for cached results,
and publishes prediction jobs to RabbitMQ.

"""

import uuid

from sqlalchemy import create_engine, text
from src.config import DB_CONN

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from pika.exceptions import AMQPError

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

engine = create_engine(
    DB_CONN,
    pool_pre_ping=True,
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
    def clean_city_name(cls, city: str) :
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
def health()  :
    """
    Return current API health status.
    """
    return {
        "status": "running"
    }


@app.post("/predict/single")
def predict_single(data: ForecastInput) :
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

        publish_prediction_job(job_message)

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
            status_code=503,
            detail=(
            "Forecast queue is currently unavailable.Please try again later."
            ),
        ) from error


@app.post("/predict/batch")
def predict_batch(
    data: list[ForecastInput],
):
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

        publish_prediction_job(job_message)

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
            status_code=503,
            detail=(
            "Forecast queue is currently unavailable.Please try again later."

            ),
        ) from error

def get_completed_result_from_database(
    job_id: str,
):
    query = text(
        """
        SELECT
            city,
            forecast_origin,
            forecast_date,
            forecast_step,
            predicted_temperature
        FROM predictions
        WHERE job_id = :job_id
        ORDER BY city, forecast_step
        """
    )

    with engine.connect() as connection:
        rows = (
            connection.execute(
                query,
                {"job_id": job_id},
            )
            .mappings()
            .all()
        )

    if not rows:
        return None

    predictions = []

    for row in rows:
        predictions.append(
            {
                "city": row["city"],
                "forecast_origin": (
                    row["forecast_origin"].isoformat()
                ),
                "forecast_date": (
                    row["forecast_date"].isoformat()
                ),
                "forecast_step": row["forecast_step"],
                "predicted_temperature": (
                    row["predicted_temperature"]
                ),
            }
        )

    response = {
        "number_of_predictions": len(predictions),
        "predictions": predictions,
        "source": "database",
        "message": (
            "Recursive forecast completed "
            "and loaded from database."
        ),
    }

    return {
        "status": "completed",
        "job_id": job_id,
        "result": response,
    }

@app.get("/predict/result/{job_id}")
def get_prediction_result(job_id: str):
    try:
        uuid.UUID(job_id)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail="Invalid job identifier.",
        ) from error

    job_status = get_job_status(job_id)

    if job_status:
        return job_status

    database_result = (
        get_completed_result_from_database(
            job_id
        )
    )

    if database_result:
        return database_result

    return {
        "status": "processing",
        "job_id": job_id,
        "message": (
            "The job is queued or processing. "
            "Check again shortly."
        ),
    }