from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
from sqlalchemy import create_engine, text

from src.config import (
    DB_CONN,
    S3_BUCKET,
    RAW_DATA_S3_KEY,
)


def read_raw_weather_csv_from_s3() -> pd.DataFrame:
    """Read the raw weather CSV from S3."""
    if not S3_BUCKET:
        raise ValueError("MODEL_S3_BUCKET is not set")

    if not RAW_DATA_S3_KEY:
        raise ValueError("RAW_DATA_S3_KEY is not set")

    print(
        f"Reading data from "
        f"s3://{S3_BUCKET}/{RAW_DATA_S3_KEY}"
    )

    s3 = boto3.client("s3")

    response = s3.get_object(
        Bucket=S3_BUCKET,
        Key=RAW_DATA_S3_KEY,
    )

    return pd.read_csv(
        BytesIO(response["Body"].read())
    )


def read_raw_weather_csv_from_local() -> pd.DataFrame:
    """Read the raw weather CSV from the local data directory."""
    current_dir = Path(__file__).resolve().parent.parent.parent

    file_path = (
        current_dir
        / "data"
        / "raw"
        / "saudi_weather_data.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file was not found: {file_path}"
        )

    print(f"Reading data from local CSV: {file_path}")

    return pd.read_csv(file_path)


def ingest():
    """Read raw weather data and insert it into PostgreSQL."""
    engine = None

    try:
        print("Reading data from CSV.........")

        # Local testing
        data = read_raw_weather_csv_from_local()

        # AWS deployment:
        # Comment the local line above and use this line.
        # data = read_raw_weather_csv_from_s3()

        print(f"Loaded {len(data)} rows")

        data["time"] = pd.to_datetime(
            data["time"],
            errors="raise",
        ).dt.date

        engine = create_engine(DB_CONN)

        print("Inserting data into raw_weather...")

        with engine.begin() as connection:
            connection.execute(
                text(
                    "TRUNCATE TABLE raw_weather "
                    "RESTART IDENTITY CASCADE;"
                )
            )

            data.to_sql(
                name="raw_weather",
                con=connection,
                if_exists="append",
                index=False,
                chunksize=1000,
                method="multi",
            )

        print("Ingestion completed successfully.")

        return data

    except Exception as error:
        raise RuntimeError(
            f"Error handling ingestion: {error}"
        ) from error

    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    ingest()