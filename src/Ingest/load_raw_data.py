import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from src.config import  DB_CONN
from io import BytesIO
import boto3

load_dotenv()



def read_raw_weather_csv_from_s3() -> pd.DataFrame:
    bucket = os.getenv("MODEL_S3_BUCKET")
    key = os.getenv("RAW_DATA_S3_KEY")

    if not bucket:
        raise ValueError("MODEL_S3_BUCKET is not set")

    if not key:
        raise ValueError("RAW_DATA_S3_KEY is not set")

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)

    return pd.read_csv(BytesIO(response["Body"].read()))

def ingest():

    try:
        print('Reading data from csv.........')
        data = read_raw_weather_csv_from_s3()

        print(f"Loaded {len(data)} rows")

        engine = create_engine(DB_CONN)

        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE raw_weather RESTART IDENTITY;"))

        data["time"] = pd.to_datetime(data["time"])
        print("Inserting data into raw_weather...")

        data.to_sql(
            name="raw_weather",
            con=engine,
            if_exists="append",
            index=False
        )
        print("Ingestion completed successfully.")

    except Exception as e:
        print(f"Error handling ingestion {str(e)}")

if __name__=="__main__":
    ingest()