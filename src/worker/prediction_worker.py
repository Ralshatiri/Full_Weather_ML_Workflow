import json
from pathlib import Path

import boto3
import joblib
import pandas as pd
import pika
from sqlalchemy import create_engine

from src.config import (
    DB_CONN,
    MODEL_PATH,
    S3_BUCKET,
    MODEL_S3_KEY,
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_QUEUE,
)

from src.services.redis_cache import (
    connect_redis,
    save_prediction_to_cache,
    save_job_status,
)


def download_model_from_s3():
    model_path = Path(MODEL_PATH)

    # Local testing: if model already exists, use it directly
    if model_path.exists():
        print(f"Using existing local model at {model_path}")
        return

    # AWS/production: download model from S3
    if not S3_BUCKET:
        raise ValueError("MODEL_S3_BUCKET is not set")

    if not MODEL_S3_KEY:
        raise ValueError("MODEL_S3_KEY is not set")

    model_path.parent.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")
    s3.download_file(S3_BUCKET, MODEL_S3_KEY, str(model_path))

    print(f"Downloaded model from s3://{S3_BUCKET}/{MODEL_S3_KEY} to {model_path}")

def process_single_prediction(job: dict, model, engine):
    job_id = job["job_id"]
    input_dict = job["input"]
    cache_key = job["cache_key"]

    input_df = pd.DataFrame([input_dict])

    prediction = round(float(model.predict(input_df)[0]), 2)

    result_df = input_df.copy()
    result_df["predicted_temperature"] = prediction

    result_df.to_sql(
        name="predictions",
        con=engine,
        if_exists="append",
        index=False
    )

    response = {
        "predicted_temperature": prediction,
        "source": "worker",
        "message": "Single prediction completed and saved to database"
    }

    save_prediction_to_cache(cache_key, response)

    save_job_status(job_id, {
        "status": "completed",
        "job_id": job_id,
        "job_type": "single_prediction",
        "result": response
    })


def process_batch_prediction(job: dict, model, engine):
    job_id = job["job_id"]
    input_list = job["input"]
    cache_key = job["cache_key"]

    input_df = pd.DataFrame(input_list)

    predictions = model.predict(input_df)

    rounded_predictions = [
        round(float(pred), 2)
        for pred in predictions
    ]

    results_df = input_df.copy()
    results_df["predicted_temperature"] = rounded_predictions

    results_df.to_sql(
        name="predictions",
        con=engine,
        if_exists="append",
        index=False
    )

    response = {
        "number_of_predictions": len(rounded_predictions),
        "predictions": rounded_predictions,
        "source": "worker",
        "message": "Batch prediction completed and saved to database"
    }

    save_prediction_to_cache(cache_key, response)

    save_job_status(job_id, {
        "status": "completed",
        "job_id": job_id,
        "job_type": "batch_prediction",
        "result": response
    })


def main():
    if not RABBITMQ_HOST:
        raise ValueError("RABBITMQ_HOST is not set")

    print("Starting prediction worker...")

    connect_redis()

    engine = create_engine(DB_CONN)

    download_model_from_s3()
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")

    credentials = pika.PlainCredentials(
        RABBITMQ_USER,
        RABBITMQ_PASSWORD
    )

    connection_params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=credentials
    )

    connection = pika.BlockingConnection(connection_params)
    channel = connection.channel()

    channel.queue_declare(
        queue=RABBITMQ_QUEUE,
        durable=True
    )

    channel.basic_qos(prefetch_count=1)

    def callback(ch, method, properties, body):
        job = None

        try:
            job = json.loads(body)
            job_id = job["job_id"]
            job_type = job["job_type"]

            print(f"Processing job {job_id} of type {job_type}")

            save_job_status(job_id, {
                "status": "processing",
                "job_id": job_id,
                "job_type": job_type,
                "message": "Prediction job is currently being processed."
            })

            if job_type == "single_prediction":
                process_single_prediction(job, model, engine)

            elif job_type == "batch_prediction":
                process_batch_prediction(job, model, engine)

            else:
                raise ValueError(f"Unknown job_type: {job_type}")

            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"Completed job {job_id}")

        except Exception as e:
            print(f"Failed to process job: {str(e)}")

            if job and "job_id" in job:
                save_job_status(job["job_id"], {
                    "status": "failed",
                    "job_id": job["job_id"],
                    "error": str(e)
                })

            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_consume(
        queue=RABBITMQ_QUEUE,
        on_message_callback=callback
    )

    print(f"Worker is waiting for jobs from queue: {RABBITMQ_QUEUE}")
    channel.start_consuming()


if __name__ == "__main__":
    main()