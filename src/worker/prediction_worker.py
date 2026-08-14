"""
RabbitMQ worker for recursive daily temperature forecasting.

The worker loads the saved LSTM and preprocessing metadata, retrieves the
latest 30 days for each requested city, recursively produces future daily
temperatures, and saves the results to PostgreSQL and Redis.
"""

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import requests

import boto3
import joblib
import numpy as np
import pandas as pd
import pika
import tensorflow as tf
from sqlalchemy import create_engine


from src.config import (
    DB_CONN,
    MODEL_METADATA_PATH,
    MODEL_METADATA_S3_KEY,
    MODEL_PATH,
    MODEL_S3_KEY,
    RABBITMQ_HOST,
    RABBITMQ_PASSWORD,
    RABBITMQ_PORT,
    RABBITMQ_QUEUE,
    RABBITMQ_USER,
    S3_BUCKET,
    OPEN_METEO_URL,
    WEATHER_TIMEZONE,
    WEATHER_HISTORY_CACHE_TTL
)

from src.services.redis_cache import (
    connect_redis,
    save_prediction_to_cache,
    save_job_status,
)

_HTTP_SESSION = requests.Session()

def download_model_from_s3():
    """
    Use existing local artifacts or download missing artifacts from S3.
    """
    artifacts = [
        (Path(MODEL_PATH), MODEL_S3_KEY),
        (Path(MODEL_METADATA_PATH), MODEL_METADATA_S3_KEY),
    ]

    missing_artifacts = [
        (local_path, s3_key)
        for local_path, s3_key in artifacts
        if not local_path.exists()
    ]

    if not missing_artifacts:
        print("Using existing local model artifacts.")
        return

    if not S3_BUCKET:
        missing_paths = [
            str(local_path)
            for local_path, _ in missing_artifacts
        ]

        raise ValueError(
            "MODEL_S3_BUCKET is not set and these local "
            f"artifacts are missing: {missing_paths}"
        )

    s3 = boto3.client("s3")

    for local_path, s3_key in missing_artifacts:
        if not s3_key:
            raise ValueError(
                f"S3 key is missing for {local_path}."
            )

        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        s3.download_file(
            S3_BUCKET,
            s3_key,
            str(local_path),
        )

        print(
            f"Downloaded s3://{S3_BUCKET}/{s3_key} "
            f"to {local_path}"
        )

def geocode_city(
    city: str
) :
    """
    Resolve a Saudi city name into coordinates.

    Returns the normalized city name, latitude, and longitude.
    """
    response = _HTTP_SESSION.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city,
            "count": 1,
            "language": "en",
            "countryCode": "SA",
            "format": "json",
        },
        timeout=15,
    )

    response.raise_for_status()

    results = response.json().get(
        "results",
        [],
    )

    if not results:
        raise ValueError(
            f"Could not find a Saudi location named '{city}'."
        )

    location = results[0]

    return (
        location["name"],
        float(location["latitude"]),
        float(location["longitude"]),
    )

def validate_recent_history(
    history_df: pd.DataFrame,
    sequence_length: int,
) :
    """Validate and prepare a recent weather sequence."""
    history_df = history_df.copy()

    history_df["time"] = pd.to_datetime(
        history_df["time"],
        errors="coerce",
    )

    history_df["temperature_2m_mean"] = pd.to_numeric(
        history_df["temperature_2m_mean"],
        errors="coerce",
    )

    history_df = (
        history_df
        .dropna(
            subset=[
                "time",
                "temperature_2m_mean",
                "latitude",
                "longitude",
            ]
        )
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .sort_values("time")
        .tail(sequence_length)
        .reset_index(drop=True)
    )

    if len(history_df) != sequence_length:
        raise ValueError(
            f"Expected {sequence_length} recent daily records, "
            f"but received {len(history_df)}."
        )

    date_differences = (
        history_df["time"]
        .diff()
        .dropna()
    )

    if not (
        date_differences == pd.Timedelta(days=1)
    ).all():
        raise ValueError(
            "Recent weather observations are not consecutive."
        )

    day_of_year = history_df["time"].dt.dayofyear

    history_df["doy_sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    history_df["doy_cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    return history_df[
        [
            "time",
            "city",
            "temperature_2m_mean",
            "doy_sin",
            "doy_cos",
            "latitude",
            "longitude",
        ]
    ]

def get_recent_city_history(
    city: str,
    sequence_length: int,
    redis_client,
) -> pd.DataFrame:
    """
    Retrieve the latest completed weather history.

    Redis allows all scaled workers to share the same downloaded history.
    """
    timezone = ZoneInfo(WEATHER_TIMEZONE)

    today = datetime.now(timezone).date()
    latest_complete_date = today - timedelta(days=1)

    cache_key = (
        f"weather_history:"
        f"{city.strip().lower()}:"
        f"{latest_complete_date.isoformat()}"
    )

    cached_value = redis_client.get(cache_key)

    if cached_value:
        print(
            f"Using cached recent history for {city}."
        )

        cached_records = json.loads(cached_value)

        return validate_recent_history(
            pd.DataFrame(cached_records),
            sequence_length,
        )

    resolved_city, latitude, longitude = (
        geocode_city(city)
    )

    print(
        f"Retrieving recent weather for "
        f"{resolved_city} from Open-Meteo."
    )

    response = _HTTP_SESSION.get(
        OPEN_METEO_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_mean",
            "past_days": sequence_length + 5,
            "forecast_days": 1,
            "timezone": WEATHER_TIMEZONE,
        },
        timeout=30,
    )

    response.raise_for_status()

    daily_data = response.json().get("daily")

    if not daily_data:
        raise ValueError(
            f"No recent weather was returned for '{resolved_city}'."
        )

    history_df = pd.DataFrame(
        {
            "time": daily_data["time"],
            "temperature_2m_mean": daily_data[
                "temperature_2m_mean"
            ],
        }
    )

    history_df["time"] = pd.to_datetime(
        history_df["time"]
    )

    # Today's daily mean is incomplete, so end on yesterday.
    history_df = history_df[
        history_df["time"].dt.date
        <= latest_complete_date
    ].copy()

    history_df["city"] = resolved_city
    history_df["latitude"] = latitude
    history_df["longitude"] = longitude

    history_df = validate_recent_history(
        history_df,
        sequence_length,
    )

    cached_df = history_df.copy()
    cached_df["time"] = (
        cached_df["time"]
        .dt.strftime("%Y-%m-%d")
    )

    redis_client.setex(
        cache_key,
        WEATHER_HISTORY_CACHE_TTL,
        json.dumps(
            cached_df.to_dict(
                orient="records"
            )
        ),
    )

    return history_df

def make_recursive_prediction(
    city: str,
    forecast_days: int,
    model,
    metadata: dict,
    redis_client,
) -> list[dict]:
    """Generate a current recursive forecast for one location."""
    sequence_length = metadata["sequence_length"]
    scaler = metadata["scaler"]
    numeric_columns = metadata["numeric_columns"]
    feature_columns = metadata["feature_columns"]
    target_column = metadata["target_column"]

    history_df = get_recent_city_history(
        city=city,
        sequence_length=sequence_length,
        redis_client=redis_client,
    )

    resolved_city = history_df["city"].iloc[-1]
    forecast_origin = history_df["time"].iloc[-1]

    raw_latitude = float(
        history_df["latitude"].iloc[-1]
    )

    raw_longitude = float(
        history_df["longitude"].iloc[-1]
    )

    # Scale all five model features.
    history_df[numeric_columns] = scaler.transform(
        history_df[numeric_columns]
    )

    window = history_df[
        feature_columns
    ].to_numpy(dtype=np.float32)

    if window.shape != (
        sequence_length,
        len(feature_columns),
    ):
        raise ValueError(
            f"Unexpected model window shape: {window.shape}"
        )

    target_index = numeric_columns.index(
        target_column
    )

    temperature_mean = scaler.mean_[target_index]
    temperature_scale = scaler.scale_[target_index]

    predictions = []

    for step in range(1, forecast_days + 1):
        predicted_scaled = float(
            model.predict(
                window[np.newaxis, ...],
                verbose=0,
            )[0][0]
        )

        predicted_temperature = (
            predicted_scaled * temperature_scale
            + temperature_mean
        )

        forecast_date = (
            forecast_origin
            + pd.Timedelta(days=step)
        )

        day_of_year = forecast_date.dayofyear

        next_numeric_df = pd.DataFrame(
            [
                {
                    target_column: predicted_temperature,
                    "doy_sin": np.sin(
                        2
                        * np.pi
                        * day_of_year
                        / 365.25
                    ),
                    "doy_cos": np.cos(
                        2
                        * np.pi
                        * day_of_year
                        / 365.25
                    ),
                    "latitude": raw_latitude,
                    "longitude": raw_longitude,
                }
            ]
        )

        next_numeric_scaled = scaler.transform(
            next_numeric_df[numeric_columns]
        )[0]

        next_values = dict(
            zip(
                numeric_columns,
                next_numeric_scaled,
            )
        )

        # Use the exact scaled prediction in the next window.
        next_values[target_column] = predicted_scaled

        next_row = np.asarray(
            [
                next_values[column]
                for column in feature_columns
            ],
            dtype=np.float32,
        )

        window = np.vstack(
            [
                window[1:],
                next_row,
            ]
        )

        predictions.append(
            {
                "city": resolved_city,
                "forecast_origin": (
                    forecast_origin.date().isoformat()
                ),
                "forecast_date": (
                    forecast_date.date().isoformat()
                ),
                "forecast_step": step,
                "predicted_temperature": round(
                    predicted_temperature,
                    2,
                ),
            }
        )

    return predictions

def process_single_prediction(
    job: dict,
    model,
    metadata: dict,
    engine,
    redis_client
):
    """
    Process one forecast job and save its result.
    """
    job_id = job["job_id"]
    input_dict = job["input"]
    cache_key = job["cache_key"]

    predictions = make_recursive_prediction(
        city=input_dict["city"],
        forecast_days=input_dict["forecast_days"],
        model=model,
        metadata=metadata,
        redis_client=redis_client,
    )

    result_df = pd.DataFrame(predictions)
    result_df["job_id"] = job_id

    result_df.to_sql(
        name="predictions",
        con=engine,
        if_exists="append",
        index=False,
    )

    response = {
        "number_of_predictions": len(predictions),
        "predictions": predictions,
        "source": "worker",
        "message": (
            "Recursive forecast completed "
            "and saved to database."
        ),
    }

    save_prediction_to_cache(
        cache_key,
        response,
    )

    save_job_status(
        job_id,
        {
            "status": "completed",
            "job_id": job_id,
            "job_type": "single_prediction",
            "result": response,
        },
    )


def process_batch_prediction(
    job: dict,
    model,
    metadata: dict,
    engine,
    redis_client
):
    """
    Process multiple forecast requests and save all results.
    """
    job_id = job["job_id"]
    input_list = job["input"]
    cache_key = job["cache_key"]

    all_predictions = []

    for input_dict in input_list:
        city_predictions = make_recursive_prediction(
            city=input_dict["city"],
            forecast_days=input_dict["forecast_days"],
            model=model,
            metadata=metadata,
            redis_client=redis_client
        )

        all_predictions.extend(
            city_predictions
        )

    results_df = pd.DataFrame(
        all_predictions
    )

    results_df["job_id"] = job_id

    results_df.to_sql(
        name="predictions",
        con=engine,
        if_exists="append",
        index=False,
    )

    response = {
        "number_of_predictions": len(
            all_predictions
        ),
        "predictions": all_predictions,
        "source": "worker",
        "message": (
            "Batch recursive forecast completed "
            "and saved to database."
        ),
    }

    save_prediction_to_cache(
        cache_key,
        response,
    )

    save_job_status(
        job_id,
        {
            "status": "completed",
            "job_id": job_id,
            "job_type": "batch_predict",
            "result": response,
        },
    )

def main():
    """
    Load the model and continuously consume prediction jobs from RabbitMQ.
    """
    if not RABBITMQ_HOST:
        raise ValueError(
            "RABBITMQ_HOST is not set."
        )

    print("Starting prediction worker...")

    redis_client = connect_redis()
    if redis_client is None:
        raise ConnectionError(
            "Worker requires Redis, but the connection failed."
        )

    engine = create_engine(DB_CONN)

    download_model_from_s3()

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    metadata = joblib.load(
        MODEL_METADATA_PATH
    )

    print(
        f"Model loaded successfully from {MODEL_PATH}"
    )

    credentials = pika.PlainCredentials(
        RABBITMQ_USER,
        RABBITMQ_PASSWORD,
    )

    connection_params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials,
    )

    connection = pika.BlockingConnection(
        connection_params
    )

    channel = connection.channel()

    channel.queue_declare(
        queue=RABBITMQ_QUEUE,
        durable=True,
    )

    channel.basic_qos(
        prefetch_count=1
    )

    def callback(ch, method, properties, body):
        """
        Process one RabbitMQ job and acknowledge the message.
        """
        job = None

        try:
            job = json.loads(body)
            job_id = job["job_id"]
            job_type = job["job_type"]

            print(
                f"Processing job {job_id} "
                f"of type {job_type}"
            )

            save_job_status(
                job_id,
                {
                    "status": "processing",
                    "job_id": job_id,
                    "job_type": job_type,
                    "message": (
                        "Prediction job is currently "
                        "being processed."
                    ),
                },
            )

            if job_type == "single_prediction":
                process_single_prediction(
                    job,
                    model,
                    metadata,
                    engine,
                    redis_client
                )

            elif job_type == "batch_predict":
                process_batch_prediction(
                    job,
                    model,
                    metadata,
                    engine,
                    redis_client
                )

            else:
                raise ValueError(
                    f"Unknown job_type: {job_type}"
                )

            ch.basic_ack(
                delivery_tag=method.delivery_tag
            )

            print(f"Completed job {job_id}")

        except Exception as error:
            print(
                f"Failed to process job: {error}"
            )

            if job and "job_id" in job:
                save_job_status(
                    job["job_id"],
                    {
                        "status": "failed",
                        "job_id": job["job_id"],
                        "error": str(error),
                    },
                )

            # Remove the invalid job without retrying forever.
            ch.basic_nack(
                delivery_tag=method.delivery_tag,
                requeue=False,
            )

    channel.basic_consume(
        queue=RABBITMQ_QUEUE,
        on_message_callback=callback,
    )

    print(
        f"Worker is waiting for jobs from queue: "
        f"{RABBITMQ_QUEUE}"
    )

    try:
        channel.start_consuming()

    finally:
        engine.dispose()

        if connection.is_open:
            connection.close()


if __name__ == "__main__":
    main()