import json
import pika

from src.config import (
    RABBITMQ_HOST,
    RABBITMQ_PORT,
    RABBITMQ_USER,
    RABBITMQ_PASSWORD,
    RABBITMQ_QUEUE,
)


def publish_prediction_job(message: dict):
    if not RABBITMQ_HOST:
        raise RuntimeError("RABBITMQ_HOST is not set. Cannot queue prediction job.")

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

    channel.basic_publish(
        exchange="",
        routing_key=RABBITMQ_QUEUE,
        body=json.dumps(message),
        properties=pika.BasicProperties(
            delivery_mode=2
        )
    )

    connection.close()

    print(f"Queued prediction job: {message.get('job_id')}")